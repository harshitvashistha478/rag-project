from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime, timezone

from src.database import get_db
from src.models.user import User
from src.schemas.user import (
    UserRegisterRequest,
    UserLoginRequest,
    RefreshTokenRequest,
    ChangePasswordRequest,
    UpdateProfileRequest,
    UserResponse,
    TokenResponse,
    AccessTokenResponse,
    MessageResponse,
)
from src.utils.security import (
    hash_password,
    verify_password,
    create_access_token,
    create_refresh_token,
    decode_token,
    get_token_expiry_seconds,
)
from src.utils.auth import get_current_active_user
import logging

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


# ─── Shared login logic ───────────────────────────────────────────────────────

def _authenticate_and_issue_tokens(email: str, password: str, db: Session) -> TokenResponse:
    """Core login logic shared by both /login (JSON) and /token (form) endpoints."""
    user = db.query(User).filter(User.email == email).first()

    if not user or not verify_password(password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated. Please contact support.",
        )

    token_data = {"sub": str(user.id), "email": user.email, "role": user.role.value}
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token({"sub": str(user.id)})

    user.refresh_token = refresh_token
    user.last_login_at = datetime.now(timezone.utc)
    db.commit()

    logger.info(f"User logged in: {user.email}")
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=get_token_expiry_seconds(),
    )


# ─── Register ────────────────────────────────────────────────────────────────

@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
)
def register(payload: UserRegisterRequest, db: Session = Depends(get_db)):
    # Check email uniqueness
    if db.query(User).filter(User.email == payload.email).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )
    # Check username uniqueness
    if db.query(User).filter(User.username == payload.username).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This username is already taken.",
        )

    user = User(
        email=payload.email,
        username=payload.username,
        full_name=payload.full_name,
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    logger.info(f"New user registered: {user.email}")
    return user


# ─── Login (JSON) ─────────────────────────────────────────────────────────────

@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Login with JSON body — returns access + refresh tokens",
)
def login(payload: UserLoginRequest, db: Session = Depends(get_db)):
    return _authenticate_and_issue_tokens(payload.email, payload.password, db)


# ─── Token (OAuth2 form) — used by Swagger UI "Authorize" button ─────────────

@router.post(
    "/token",
    response_model=TokenResponse,
    summary="OAuth2-compatible login (form data) — for Swagger UI authorization",
    include_in_schema=True,
)
def login_oauth2_form(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """
    Accepts application/x-www-form-urlencoded with fields:
      - username  (we treat this as the email)
      - password
    This endpoint is what FastAPI's Swagger 'Authorize' button posts to.
    """
    return _authenticate_and_issue_tokens(form_data.username, form_data.password, db)





# ─── Refresh Token ───────────────────────────────────────────────────────────

@router.post(
    "/refresh",
    response_model=AccessTokenResponse,
    summary="Get a new access token using a valid refresh token",
)
def refresh_access_token(payload: RefreshTokenRequest, db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired refresh token.",
        headers={"WWW-Authenticate": "Bearer"},
    )

    token_payload = decode_token(payload.refresh_token)
    if token_payload is None or token_payload.get("type") != "refresh":
        raise credentials_exception

    from uuid import UUID
    try:
        user_id = UUID(token_payload.get("sub"))
    except (ValueError, TypeError):
        raise credentials_exception

    user = db.query(User).filter(User.id == user_id).first()
    if not user or not user.is_active:
        raise credentials_exception

    # Validate token matches what's stored (prevents reuse after logout)
    if user.refresh_token != payload.refresh_token:
        raise credentials_exception

    token_data = {"sub": str(user.id), "email": user.email, "role": user.role.value}
    access_token = create_access_token(token_data)

    return AccessTokenResponse(
        access_token=access_token,
        expires_in=get_token_expiry_seconds(),
    )


# ─── Logout ──────────────────────────────────────────────────────────────────

@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Logout — invalidates the stored refresh token",
)
def logout(
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    current_user.refresh_token = None
    db.commit()
    logger.info(f"User logged out: {current_user.email}")
    return MessageResponse(message="Successfully logged out.")


# ─── Get Current User (Me) ───────────────────────────────────────────────────

@router.get(
    "/me",
    response_model=UserResponse,
    summary="Get the currently authenticated user's profile",
)
def get_me(current_user: User = Depends(get_current_active_user)):
    return current_user


# ─── Update Profile ──────────────────────────────────────────────────────────

@router.patch(
    "/me",
    response_model=UserResponse,
    summary="Update the current user's profile",
)
def update_profile(
    payload: UpdateProfileRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    if payload.username and payload.username != current_user.username:
        if db.query(User).filter(User.username == payload.username).first():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This username is already taken.",
            )
        current_user.username = payload.username

    if payload.full_name is not None:
        current_user.full_name = payload.full_name

    db.commit()
    db.refresh(current_user)
    return current_user


# ─── Change Password ─────────────────────────────────────────────────────────

@router.post(
    "/change-password",
    response_model=MessageResponse,
    summary="Change the current user's password",
)
def change_password(
    payload: ChangePasswordRequest,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db),
):
    if not verify_password(payload.current_password, current_user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Current password is incorrect.",
        )

    if payload.current_password == payload.new_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be different from the current password.",
        )

    current_user.hashed_password = hash_password(payload.new_password)
    # Invalidate refresh token — forces re-login on all devices
    current_user.refresh_token = None
    db.commit()

    logger.info(f"Password changed for user: {current_user.email}")
    return MessageResponse(message="Password updated successfully. Please log in again.")