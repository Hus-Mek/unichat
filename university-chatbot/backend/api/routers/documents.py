"""Document upload and management endpoints."""

from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.access_control import AccessLevel
from ...db.database import get_db
from ...models.user import User
from ...rag.vector_store import VectorStore
from ...schemas.document import DocumentListResponse, DocumentResponse
from ...services.collection_service import get_collection
from ...services.document_service import (
    delete_document,
    get_document,
    list_documents,
    upload_and_index,
)
from ..deps import get_current_user, require_access_level

router = APIRouter()

_vector_store: VectorStore | None = None


def _get_vs() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


@router.post("/upload", response_model=DocumentResponse, status_code=201)
async def upload_document(
    file: UploadFile = File(...),
    collection_id: UUID = Form(...),
    access_level: str = Form("public"),
    owner_id: UUID | None = Form(None),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload and index a document."""
    collection = await get_collection(db, collection_id)
    doc = await upload_and_index(
        db=db,
        file=file,
        collection_id=collection_id,
        collection_chroma_name=collection.chroma_collection_name,
        access_level=access_level,
        uploaded_by=user.id,
        owner_id=owner_id,
        vector_store=_get_vs(),
    )
    return DocumentResponse.model_validate(doc)


@router.get("/", response_model=DocumentListResponse)
async def list_docs(
    collection_id: UUID | None = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List documents, optionally filtered by collection."""
    docs = await list_documents(db, collection_id)
    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(d) for d in docs],
        total=len(docs),
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_doc(
    document_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get document metadata."""
    doc = await get_document(db, document_id)
    return DocumentResponse.model_validate(doc)


@router.delete(
    "/{document_id}",
    status_code=204,
    dependencies=[Depends(require_access_level(AccessLevel.ADMIN_STAFF))],
)
async def delete_doc(
    document_id: UUID,
    db: AsyncSession = Depends(get_db),
):
    """Delete a document (admin staff and above)."""
    await delete_document(db, document_id, _get_vs())
