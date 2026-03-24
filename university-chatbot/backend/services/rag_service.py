"""RAG orchestration service: retrieval + LLM query."""

import time
from uuid import UUID, uuid4

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.access_control import AccessLevel, parse_access_level
from ..models.collection import Collection
from ..models.session import ChatMessage, ChatSession
from ..rag.retriever import Retriever
from ..rag.vector_store import VectorStore
from ..schemas.chat import ChatRequest, ChatResponse, SourceInfo, TokenUsage
from .llm_service import LLMService


async def handle_query(
    db: AsyncSession,
    request: ChatRequest,
    user_id: UUID,
    user_level_str: str,
    vector_store: VectorStore,
    llm_service: LLMService,
) -> ChatResponse:
    """Execute a RAG query: retrieve context, call LLM, log results."""
    start = time.monotonic()
    user_level = parse_access_level(user_level_str)

    # Determine which collections to search
    if request.collection_ids:
        stmt = select(Collection).where(
            Collection.id.in_(request.collection_ids),
            Collection.is_active.is_(True),
        )
    else:
        stmt = select(Collection).where(Collection.is_active.is_(True))

    result = await db.execute(stmt)
    collections = list(result.scalars().all())

    col_names = [c.chroma_collection_name for c in collections]
    col_min_levels = {
        c.chroma_collection_name: c.min_access_level for c in collections
    }

    # Retrieve context
    retriever = Retriever(vector_store)
    retrieval = retriever.retrieve(
        question=request.question,
        user_level=user_level,
        collection_names=col_names,
        collection_min_levels=col_min_levels,
        user_id=str(user_id),
    )

    context = retrieval["context"]
    sources_detail = retrieval["sources_detail"]
    source_names = list(sources_detail.keys())

    # Call LLM
    model = request.model or llm_service._settings.DEFAULT_MODEL
    llm_result = llm_service.query(
        question=request.question,
        context=context,
        sources=source_names,
        model=model,
        max_tokens=request.max_tokens,
    )

    elapsed_ms = int((time.monotonic() - start) * 1000)

    # Calculate cost
    cost_info = LLMService.calculate_cost(llm_result.get("tokens"), model)

    # Build source info
    sources = [
        SourceInfo(file_name=name, access_level=info["access_level"], chunks=info["count"])
        for name, info in sources_detail.items()
    ]

    # Build token usage
    tokens = None
    raw_tokens = llm_result.get("tokens")
    if raw_tokens:
        tokens = TokenUsage(
            prompt=raw_tokens["prompt"],
            completion=raw_tokens["completion"],
            total=raw_tokens["total"],
        )

    # Get or create chat session
    session_id = request.session_id
    if not session_id:
        session = ChatSession(
            user_id=user_id,
            collection_id=request.collection_ids[0] if request.collection_ids else None,
            title=request.question[:100],
        )
        db.add(session)
        await db.flush()
        await db.refresh(session)
        session_id = session.id

    # Save user message
    user_msg = ChatMessage(
        session_id=session_id,
        role="user",
        content=request.question,
    )
    db.add(user_msg)

    # Save assistant message
    msg_id = uuid4()
    assistant_msg = ChatMessage(
        id=msg_id,
        session_id=session_id,
        role="assistant",
        content=llm_result["text"],
        model_used=model,
        tokens_prompt=raw_tokens["prompt"] if raw_tokens else None,
        tokens_completion=raw_tokens["completion"] if raw_tokens else None,
        tokens_total=raw_tokens["total"] if raw_tokens else None,
        cost_usd=cost_info["total_cost"],
        retrieval_count=retrieval["count"],
        sources={name: info for name, info in sources_detail.items()},
        response_time_ms=elapsed_ms,
        finish_reason=llm_result.get("finish_reason"),
    )
    db.add(assistant_msg)
    await db.flush()

    return ChatResponse(
        answer=llm_result["text"],
        sources=sources,
        tokens=tokens,
        cost_usd=cost_info["total_cost"],
        model_used=model,
        response_time_ms=elapsed_ms,
        session_id=session_id,
        message_id=msg_id,
    )
