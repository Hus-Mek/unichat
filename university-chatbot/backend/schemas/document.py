from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime

class DocumentResponse(BaseModel):
    id: UUID
    collection_id: UUID
    file_name: str
    file_type: str
    file_size_bytes: int | None
    access_level: str
    owner_id: UUID | None
    chunk_count: int
    status: str
    uploaded_by: UUID
    created_at: datetime

    model_config = {"from_attributes": True}

class DocumentUploadRequest(BaseModel):
    """Metadata sent alongside file upload (as form fields)"""
    collection_id: UUID
    access_level: str = "public"
    owner_id: UUID | None = None

class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int
