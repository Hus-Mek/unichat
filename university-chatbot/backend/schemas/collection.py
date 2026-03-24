from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime

class CollectionCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    department: str | None = None
    min_access_level: str = "public"

class CollectionUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    department: str | None = None
    min_access_level: str | None = None
    is_active: bool | None = None

class CollectionResponse(BaseModel):
    id: UUID
    name: str
    slug: str
    description: str | None
    department: str | None
    min_access_level: str
    is_active: bool
    document_count: int = 0
    created_at: datetime

    model_config = {"from_attributes": True}

class GrantAccessRequest(BaseModel):
    user_id: UUID
