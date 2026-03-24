"""Document upload, processing, and indexing service."""

import hashlib
import os
from uuid import UUID

from fastapi import HTTPException, UploadFile
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.config import get_settings
from ..models.document import Document, DocumentChunk
from ..rag.chunker import chunk_text
from ..rag.document_processor import SUPPORTED_EXTENSIONS, process_document
from ..rag.vector_store import VectorStore


async def upload_and_index(
    db: AsyncSession,
    file: UploadFile,
    collection_id: UUID,
    collection_chroma_name: str,
    access_level: str,
    uploaded_by: UUID,
    owner_id: UUID | None,
    vector_store: VectorStore,
) -> Document:
    """Upload a file, extract text, chunk, embed, and store in ChromaDB + PostgreSQL."""
    file_bytes = await file.read()
    file_name = file.filename or "unknown"
    file_ext = os.path.splitext(file_name)[1].lower()

    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file_ext}. Supported: {SUPPORTED_EXTENSIONS}",
        )

    file_hash = hashlib.sha256(file_bytes).hexdigest()
    settings = get_settings()

    # Create document record
    doc = Document(
        collection_id=collection_id,
        file_name=file_name,
        file_type=file_ext.lstrip("."),
        file_size_bytes=len(file_bytes),
        file_hash=file_hash,
        access_level=access_level,
        owner_id=owner_id,
        uploaded_by=uploaded_by,
        status="processing",
    )
    db.add(doc)
    await db.flush()
    await db.refresh(doc)

    try:
        # Extract text
        text = process_document(file_bytes, file_ext)
        if not text.strip():
            doc.status = "failed"
            doc.error_message = "No text extracted"
            await db.flush()
            raise HTTPException(status_code=400, detail="No text could be extracted from the document")

        # Chunk
        chunks = chunk_text(text, settings.CHUNK_SIZE, settings.CHUNK_OVERLAP)
        if not chunks:
            doc.status = "failed"
            doc.error_message = "Chunking produced no results"
            await db.flush()
            raise HTTPException(status_code=400, detail="Document could not be chunked")

        # Embed
        embeddings = vector_store.embed(chunks)

        # Build IDs and metadata for ChromaDB
        chunk_ids = []
        chroma_metadatas = []
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc.id}_{i}"
            chunk_ids.append(chunk_id)
            chroma_metadatas.append({
                "access_level": access_level,
                "source": file_name,
                "chunk_index": i,
                "owner_id": str(owner_id) if owner_id else "",
                "document_id": str(doc.id),
            })

            # Mirror in PostgreSQL
            db_chunk = DocumentChunk(
                document_id=doc.id,
                chroma_id=chunk_id,
                chunk_index=i,
                chunk_text=chunk,
                char_count=len(chunk),
            )
            db.add(db_chunk)

        # Store in ChromaDB
        vector_store.add_documents(
            collection_name=collection_chroma_name,
            ids=chunk_ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=chroma_metadatas,
        )

        doc.chunk_count = len(chunks)
        doc.status = "indexed"
        await db.flush()
        await db.refresh(doc)
        return doc

    except HTTPException:
        raise
    except Exception as e:
        doc.status = "failed"
        doc.error_message = str(e)
        await db.flush()
        raise HTTPException(status_code=500, detail=f"Indexing failed: {e}")


async def list_documents(
    db: AsyncSession, collection_id: UUID | None = None
) -> list[Document]:
    """List documents, optionally filtered by collection."""
    stmt = select(Document).where(Document.status != "deleted")
    if collection_id:
        stmt = stmt.where(Document.collection_id == collection_id)
    stmt = stmt.order_by(Document.created_at.desc())
    result = await db.execute(stmt)
    return list(result.scalars().all())


async def get_document(db: AsyncSession, document_id: UUID) -> Document:
    """Get a document by ID."""
    result = await db.execute(select(Document).where(Document.id == document_id))
    doc = result.scalar_one_or_none()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


async def delete_document(
    db: AsyncSession, document_id: UUID, vector_store: VectorStore
) -> None:
    """Mark a document as deleted and remove its chunks from ChromaDB."""
    doc = await get_document(db, document_id)

    # Get chunk IDs to remove from ChromaDB
    result = await db.execute(
        select(DocumentChunk.chroma_id).where(DocumentChunk.document_id == doc.id)
    )
    chroma_ids = [row[0] for row in result.all()]

    if chroma_ids:
        # Get the collection's chroma name
        from ..models.collection import Collection
        col_result = await db.execute(
            select(Collection.chroma_collection_name).where(Collection.id == doc.collection_id)
        )
        chroma_name = col_result.scalar_one_or_none()
        if chroma_name:
            try:
                collection = vector_store.get_or_create_collection(chroma_name)
                collection.delete(ids=chroma_ids)
            except Exception:
                pass

    doc.status = "deleted"
    await db.flush()
