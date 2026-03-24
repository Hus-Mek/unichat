from pydantic import BaseModel, Field
from uuid import UUID
from datetime import datetime

class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=5000)
    collection_ids: list[UUID] | None = None  # null = all accessible
    model: str | None = None
    max_tokens: int = Field(default=2048, ge=100, le=8192)
    session_id: UUID | None = None  # null = create new session

class TokenUsage(BaseModel):
    prompt: int
    completion: int
    total: int

class SourceInfo(BaseModel):
    file_name: str
    access_level: str
    chunks: int

class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceInfo]
    tokens: TokenUsage | None
    cost_usd: float | None
    model_used: str
    response_time_ms: int
    session_id: UUID
    message_id: UUID

class ChatSessionResponse(BaseModel):
    id: UUID
    title: str | None
    collection_id: UUID | None
    is_active: bool
    created_at: datetime
    message_count: int = 0

    model_config = {"from_attributes": True}

class ChatMessageResponse(BaseModel):
    id: UUID
    role: str
    content: str
    model_used: str | None
    tokens: TokenUsage | None
    cost_usd: float | None
    sources: list[SourceInfo] | None
    response_time_ms: int | None
    created_at: datetime

    model_config = {"from_attributes": True}
