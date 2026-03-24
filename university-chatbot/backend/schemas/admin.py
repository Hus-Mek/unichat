from pydantic import BaseModel
from uuid import UUID
from datetime import datetime
from typing import Any

class AuditLogResponse(BaseModel):
    id: UUID
    user_id: UUID | None
    action: str
    resource_type: str | None
    resource_id: UUID | None
    details: dict[str, Any] | None
    ip_address: str | None
    created_at: datetime

    model_config = {"from_attributes": True}

class SystemStats(BaseModel):
    total_users: int
    total_documents: int
    total_collections: int
    total_queries: int
    total_chunks: int

class UsageStats(BaseModel):
    period: str
    queries: int
    tokens_used: int
    cost_usd: float
    unique_users: int

class ReportRequest(BaseModel):
    university_name: str = "University"
    expected_daily_queries: int = 1000
    preferred_deployment: str | None = None  # onprem | cloud | hybrid
