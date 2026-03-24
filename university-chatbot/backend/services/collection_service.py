"""Collection (multi-tenant document space) management service."""

import re
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.collection import Collection, UserCollectionAccess
from ..models.document import Document
from ..schemas.collection import CollectionCreate, CollectionResponse, CollectionUpdate


def _slugify(name: str) -> str:
    """Generate a URL-safe slug from a name."""
    slug = name.lower().strip()
    slug = re.sub(r"[^\w\s-]", "", slug)
    slug = re.sub(r"[\s_]+", "-", slug)
    return slug.strip("-")


async def create_collection(
    db: AsyncSession, data: CollectionCreate, created_by: UUID
) -> Collection:
    """Create a new document collection."""
    slug = _slugify(data.name)

    # Check for duplicate slug
    existing = await db.execute(select(Collection).where(Collection.slug == slug))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Collection with this name already exists")

    chroma_name = f"col_{slug.replace('-', '_')}"

    collection = Collection(
        name=data.name,
        slug=slug,
        description=data.description,
        department=data.department,
        min_access_level=data.min_access_level,
        chroma_collection_name=chroma_name,
        created_by=created_by,
    )
    db.add(collection)
    await db.flush()
    await db.refresh(collection)
    return collection


async def list_collections(db: AsyncSession) -> list[CollectionResponse]:
    """List all active collections with document counts."""
    stmt = (
        select(
            Collection,
            func.count(Document.id).label("doc_count"),
        )
        .outerjoin(Document, Document.collection_id == Collection.id)
        .where(Collection.is_active.is_(True))
        .group_by(Collection.id)
    )
    results = await db.execute(stmt)
    rows = results.all()

    return [
        CollectionResponse(
            id=col.id,
            name=col.name,
            slug=col.slug,
            description=col.description,
            department=col.department,
            min_access_level=col.min_access_level,
            is_active=col.is_active,
            document_count=doc_count,
            created_at=col.created_at,
        )
        for col, doc_count in rows
    ]


async def get_collection(db: AsyncSession, collection_id: UUID) -> Collection:
    """Get a single collection by ID."""
    result = await db.execute(
        select(Collection).where(Collection.id == collection_id)
    )
    collection = result.scalar_one_or_none()
    if not collection:
        raise HTTPException(status_code=404, detail="Collection not found")
    return collection


async def update_collection(
    db: AsyncSession, collection_id: UUID, data: CollectionUpdate
) -> Collection:
    """Update a collection's mutable fields."""
    collection = await get_collection(db, collection_id)
    for field, value in data.model_dump(exclude_unset=True).items():
        setattr(collection, field, value)
    await db.flush()
    await db.refresh(collection)
    return collection


async def delete_collection(db: AsyncSession, collection_id: UUID) -> None:
    """Soft-delete (deactivate) a collection."""
    collection = await get_collection(db, collection_id)
    collection.is_active = False
    await db.flush()


async def grant_user_access(
    db: AsyncSession,
    collection_id: UUID,
    user_id: UUID,
    granted_by: UUID,
) -> None:
    """Grant a user explicit access to a collection."""
    access = UserCollectionAccess(
        user_id=user_id,
        collection_id=collection_id,
        granted_by=granted_by,
    )
    db.add(access)
    await db.flush()
