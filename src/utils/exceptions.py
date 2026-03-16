from fastapi import HTTPException, status


class UserNotFoundException(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found.",
        )


class UserAlreadyExistsException(HTTPException):
    def __init__(self, field: str = "email"):
        super().__init__(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"A user with this {field} already exists.",
        )


class InvalidCredentialsException(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )


class InactiveAccountException(HTTPException):
    def __init__(self):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated. Please contact support.",
        )


class InvalidTokenException(HTTPException):
    def __init__(self, detail: str = "Invalid or expired token."):
        super().__init__(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"},
        )


class PermissionDeniedException(HTTPException):
    def __init__(self, detail: str = "You do not have permission to perform this action."):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=detail,
        )