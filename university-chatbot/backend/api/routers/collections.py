"""Collection (multi-tenant document space) endpoints."""

from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.access_control import AccessLevel
from ...db.database import get_db
from ...models.user import User
from ...schemas.collection import (
    CollectionCreate,
    CollectionResponse,
    CollectionUpdate,
    GrantAccessRequest,
)
from ...services.collection_service import (
    create_collection,
    delete_collection,
    get_collection,
    grant_user_access,
    list_collections,
    update_collection,
)
from ..deps import get_current_user, require_access_level

router = APIRouter()


@router.post(
    "/",
    response_model=CollectionResponse,
    status_code=201,
    dependencies=[Depends(require_access_level(AccessLevel.ADMIN_STAFF))],
)
async def create(
    data: CollectionCreate,
    user: User = Depends(require_access_level(AccessLevel.ADMIN_STAFF)),
    db: AsyncSession = Depends(get_db),
):
    """Create a new document collection (admin staff+)."""
    col = await create_collection(db, data, user.id)
    return CollectionResponse(
        id=col.id,
        name=col.name,
        slug=col.slug,
        description=col.description,
        department=col.department,
        min_access_level=col.min_access_level,
        is_active=col.is_active,
        document_count=0,
        created_at=col.created_at,
    )


@router.get("/", response_model=list[CollectionResponse])
async def list_all(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all active collections."""
    return await list_collections(db)


@router.get("/{collection_id}", response_model=CollectionResponse)
async def get_one(
    collection_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get a collection by ID."""
    col = await get_collection(db, collection_id)
    return CollectionResponse(
        id=col.id,
        name=col.name,
        slug=col.slug,
        description=col.description,
        department=col.department,
        min_access_level=col.min_access_level,
        is_active=col.is_active,
        document_count=0,
        created_at=col.created_at,
    )


@router.put(
    "/{collection_id}",
    response_model=CollectionResponse,
    dependencies=[Depends(require_access_level(AccessLevel.ADMIN_STAFF))],
)
async def update(
    collection_id: UUID,
    data: CollectionUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update collection settings (admin staff+)."""
    col = await update_collection(db, collection_id, data)
    return CollectionResponse(
        id=col.id,
        name=col.name,
        slug=col.slug,
        description=col.description,
        department=col.department,
        min_access_level=col.min_access_level,
        is_active=col.is_active,
        document_count=0,
        created_at=col.created_at,
    )


@router.delete(
    "/{collection_id}",
    status_code=204,
    dependencies=[Depends(require_access_level(AccessLevel.EXECUTIVE_BOARD))],
)
async def delete(
    collection_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Delete a collection (executive board only)."""
    await delete_collection(db, collection_id)


@router.post(
    "/{collection_id}/grant-access",
    status_code=204,
    dependencies=[Depends(require_access_level(AccessLevel.ADMIN_STAFF))],
)
async def grant_access(
    collection_id: UUID,
    data: GrantAccessRequest,
    user: User = Depends(require_access_level(AccessLevel.ADMIN_STAFF)),
    db: AsyncSession = Depends(get_db),
):
    """Grant a user access to a collection (admin staff+)."""
    await grant_user_access(db, collection_id, data.user_id, user.id)
