from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from uuid import UUID
from src.database import get_db
from src.models.user import User, UserRole
from src.utils.security import decode_token
import logging

logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")


def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    FastAPI dependency: extracts and validates the Bearer token,
    then loads and returns the corresponding User from the DB.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    payload = decode_token(token)
    if payload is None:
        raise credentials_exception

    # Ensure this is an access token, not a refresh token
    if payload.get("type") != "access":
        raise credentials_exception

    user_id: str = payload.get("sub")
    if user_id is None:
        raise credentials_exception

    try:
        uid = UUID(user_id)
    except ValueError:
        raise credentials_exception

    user = db.query(User).filter(User.id == uid).first()
    if user is None:
        raise credentials_exception

    return user


def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Dependency: additionally ensures the user account is active."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated. Please contact support.",
        )
    return current_user


def require_role(*roles: UserRole):
    """
    Dependency factory for role-based access control.

    Usage:
        @router.get("/admin", dependencies=[Depends(require_role(UserRole.ADMIN))])
    """
    def role_checker(current_user: User = Depends(get_current_active_user)) -> User:
        if current_user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required role(s): {[r.value for r in roles]}",
            )
        return current_user
    return role_checker


# Convenience shorthand dependencies
require_admin = require_role(UserRole.ADMIN)