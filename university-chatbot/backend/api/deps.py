"""FastAPI dependency injection for authentication and database access."""

from uuid import UUID

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.access_control import AccessLevel, parse_access_level
from ..core.config import get_settings
from ..core.security import verify_token
from ..db.database import get_db
from ..models.user import User

security_scheme = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Extract and validate the current user from the JWT bearer token."""
    try:
        payload = verify_token(credentials.credentials)
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401, detail="Invalid token payload")
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    result = await db.execute(select(User).where(User.id == UUID(user_id)))
    user = result.scalar_one_or_none()

    if user is None or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    return user


def require_access_level(min_level: AccessLevel):
    """
    Dependency factory that enforces a minimum access level.

    Usage:
        @router.get("/admin", dependencies=[Depends(require_access_level(AccessLevel.ADMIN_STAFF))])
    """
    async def _checker(user: User = Depends(get_current_user)) -> User:
        user_level = parse_access_level(user.access_level)
        if user_level < min_level:
            raise HTTPException(
                status_code=403,
                detail=f"Requires {min_level.name.lower()} access or higher",
            )
        return user
    return _checker
