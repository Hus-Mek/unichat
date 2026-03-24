"""Multi-collection ChromaDB vector store."""

import chromadb
from sentence_transformers import SentenceTransformer

from ..core.config import get_settings


class VectorStore:
    """Wraps ChromaDB with support for multiple named collections."""

    def __init__(self):
        settings = get_settings()
        self._client = chromadb.PersistentClient(path=settings.CHROMA_DIR)
        self._embedder = SentenceTransformer(settings.EMBED_MODEL)

    def get_or_create_collection(self, name: str) -> chromadb.Collection:
        return self._client.get_or_create_collection(name)

    def delete_collection(self, name: str) -> None:
        self._client.delete_collection(name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.encode([text])[0].tolist()

    def add_documents(
        self,
        collection_name: str,
        ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> None:
        collection = self.get_or_create_collection(collection_name)
        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    def query(
        self,
        collection_name: str,
        query_embedding: list[float],
        n_results: int = 15,
        where: dict | None = None,
    ) -> dict:
        collection = self.get_or_create_collection(collection_name)
        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
        }
        if where:
            kwargs["where"] = where
        return collection.query(**kwargs)

    def get_collection_count(self, collection_name: str) -> int:
        collection = self.get_or_create_collection(collection_name)
        return collection.count()

    def clear_collection(self, collection_name: str) -> int:
        collection = self.get_or_create_collection(collection_name)
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)
        return len(all_ids)
