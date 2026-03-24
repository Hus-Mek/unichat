"""RAG retriever with access control and multi-collection support."""

from ..core.access_control import (
    AccessLevel,
    build_chroma_filter,
    can_access_collection,
    parse_access_level,
)
from ..core.config import get_settings
from .vector_store import VectorStore


class Retriever:
    """Retrieves relevant document chunks with access control."""

    def __init__(self, vector_store: VectorStore):
        self._vs = vector_store
        self._settings = get_settings()

    def retrieve(
        self,
        question: str,
        user_level: AccessLevel,
        collection_names: list[str],
        collection_min_levels: dict[str, str],
        user_id: str | None = None,
        n_results: int | None = None,
    ) -> dict:
        """
        Retrieve relevant chunks across multiple collections.

        Returns dict with: documents, context, sources_detail, count, metadatas.
        """
        if n_results is None:
            n_results = self._settings.DEFAULT_N_RESULTS

        query_embedding = self._vs.embed_query(question)
        where_filter = build_chroma_filter(user_level, user_id)

        all_documents: list[str] = []
        all_metadatas: list[dict] = []

        for col_name in collection_names:
            min_level_str = collection_min_levels.get(col_name, "public")
            min_level = parse_access_level(min_level_str)
            if not can_access_collection(user_level, min_level):
                continue

            try:
                results = self._vs.query(
                    collection_name=col_name,
                    query_embedding=query_embedding,
                    n_results=n_results,
                    where=where_filter,
                )
                docs = results.get("documents", [[]])[0]
                metas = results.get("metadatas", [[]])[0]
                all_documents.extend(docs)
                all_metadatas.extend(metas)
            except Exception:
                continue

        if not all_documents:
            return {
                "documents": [],
                "context": "",
                "sources_detail": {},
                "count": 0,
                "metadatas": [],
            }

        context = "\n\n".join(all_documents)

        sources_detail: dict[str, dict] = {}
        for meta in all_metadatas:
            source = meta.get("source", "unknown")
            access = meta.get("access_level", "unknown")
            if source not in sources_detail:
                sources_detail[source] = {"count": 0, "access_level": access}
            sources_detail[source]["count"] += 1

        return {
            "documents": all_documents,
            "context": context,
            "sources_detail": sources_detail,
            "count": len(all_documents),
            "metadatas": all_metadatas,
        }
