"""Chat endpoints: RAG query and session management."""

from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from ...db.database import get_db
from ...models.session import ChatMessage, ChatSession
from ...models.user import User
from ...rag.vector_store import VectorStore
from ...schemas.chat import ChatMessageResponse, ChatRequest, ChatResponse, ChatSessionResponse, TokenUsage
from ...services.llm_service import LLMService
from ...services.rag_service import handle_query
from ..deps import get_current_user

router = APIRouter()

# Singletons initialised at first use
_vector_store: VectorStore | None = None
_llm_service: LLMService | None = None


def _get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is None:
        _vector_store = VectorStore()
    return _vector_store


def _get_llm_service() -> LLMService:
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service


@router.post("/query", response_model=ChatResponse)
async def query(
    request: ChatRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Send a question and get a RAG-powered answer."""
    return await handle_query(
        db=db,
        request=request,
        user_id=user.id,
        user_level_str=user.access_level,
        vector_store=_get_vector_store(),
        llm_service=_get_llm_service(),
    )


@router.get("/sessions", response_model=list[ChatSessionResponse])
async def list_sessions(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List the current user's chat sessions."""
    stmt = (
        select(
            ChatSession,
            func.count(ChatMessage.id).label("msg_count"),
        )
        .outerjoin(ChatMessage, ChatMessage.session_id == ChatSession.id)
        .where(ChatSession.user_id == user.id, ChatSession.is_active.is_(True))
        .group_by(ChatSession.id)
        .order_by(ChatSession.created_at.desc())
    )
    results = await db.execute(stmt)
    return [
        ChatSessionResponse(
            id=session.id,
            title=session.title,
            collection_id=session.collection_id,
            is_active=session.is_active,
            created_at=session.created_at,
            message_count=msg_count,
        )
        for session, msg_count in results.all()
    ]


@router.get("/sessions/{session_id}/messages", response_model=list[ChatMessageResponse])
async def get_session_messages(
    session_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get all messages in a chat session."""
    # Verify session belongs to user
    session_result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id, ChatSession.user_id == user.id
        )
    )
    session = session_result.scalar_one_or_none()
    if not session:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Session not found")

    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
    )
    messages = result.scalars().all()
    out = []
    for msg in messages:
        tokens = None
        if msg.tokens_prompt is not None:
            tokens = TokenUsage(
                prompt=msg.tokens_prompt,
                completion=msg.tokens_completion or 0,
                total=msg.tokens_total or 0,
            )
        out.append(ChatMessageResponse(
            id=msg.id,
            role=msg.role,
            content=msg.content,
            model_used=msg.model_used,
            tokens=tokens,
            cost_usd=float(msg.cost_usd) if msg.cost_usd else None,
            sources=None,
            response_time_ms=msg.response_time_ms,
            created_at=msg.created_at,
        ))
    return out


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: UUID,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Deactivate a chat session."""
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id, ChatSession.user_id == user.id
        )
    )
    session = result.scalar_one_or_none()
    if session:
        session.is_active = False
        await db.flush()
