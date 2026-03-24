"""Admin endpoints: user management, audit logs, stats, report generation."""

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.access_control import AccessLevel
from ...db.database import get_db
from ...models.audit import AuditLog
from ...models.document import Document, DocumentChunk
from ...models.session import ChatMessage
from ...models.user import User
from ...schemas.admin import AuditLogResponse, ReportRequest, SystemStats
from ...schemas.auth import UserResponse, UserUpdate
from ..deps import require_access_level

router = APIRouter()


@router.get(
    "/users",
    response_model=list[UserResponse],
    dependencies=[Depends(require_access_level(AccessLevel.ADMIN_STAFF))],
)
async def list_users(db: AsyncSession = Depends(get_db)):
    """List all users."""
    result = await db.execute(select(User).order_by(User.created_at.desc()))
    return [UserResponse.model_validate(u) for u in result.scalars().all()]


@router.put(
    "/users/{user_id}",
    response_model=UserResponse,
    dependencies=[Depends(require_access_level(AccessLevel.ADMIN_STAFF))],
)
async def update_user(
    user_id: UUID,
    data: UserUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update a user's profile or access level (admin staff+)."""
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    for field, value in data.model_dump(exclude_unset=True).items():
        setattr(user, field, value)
    await db.flush()
    await db.refresh(user)
    return UserResponse.model_validate(user)


@router.get(
    "/audit-logs",
    response_model=list[AuditLogResponse],
    dependencies=[Depends(require_access_level(AccessLevel.ADMIN_STAFF))],
)
async def get_audit_logs(
    limit: int = 100,
    offset: int = 0,
    db: AsyncSession = Depends(get_db),
):
    """Query audit logs (admin staff+)."""
    stmt = (
        select(AuditLog)
        .order_by(AuditLog.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await db.execute(stmt)
    return [AuditLogResponse.model_validate(a) for a in result.scalars().all()]


@router.get(
    "/stats",
    response_model=SystemStats,
    dependencies=[Depends(require_access_level(AccessLevel.ADMIN_STAFF))],
)
async def get_stats(db: AsyncSession = Depends(get_db)):
    """System-wide statistics (admin staff+)."""
    users = await db.execute(select(func.count(User.id)))
    docs = await db.execute(select(func.count(Document.id)))
    chunks = await db.execute(select(func.count(DocumentChunk.id)))
    queries = await db.execute(
        select(func.count(ChatMessage.id)).where(ChatMessage.role == "user")
    )

    from ...models.collection import Collection
    cols = await db.execute(select(func.count(Collection.id)))

    return SystemStats(
        total_users=users.scalar() or 0,
        total_documents=docs.scalar() or 0,
        total_collections=cols.scalar() or 0,
        total_queries=queries.scalar() or 0,
        total_chunks=chunks.scalar() or 0,
    )


@router.post(
    "/generate-report",
    dependencies=[Depends(require_access_level(AccessLevel.EXECUTIVE_BOARD))],
)
async def generate_report(data: ReportRequest):
    """Generate a PowerPoint proposal (executive board only)."""
    from ...utils.pptx_generator import generate_proposal

    pptx_bytes = generate_proposal(
        university_name=data.university_name,
        expected_daily_queries=data.expected_daily_queries,
        preferred_deployment=data.preferred_deployment,
    )
    return StreamingResponse(
        pptx_bytes,
        media_type="application/vnd.openxmlformats-officedocument.presentationml.presentation",
        headers={"Content-Disposition": "attachment; filename=university_chatbot_proposal.pptx"},
    )
