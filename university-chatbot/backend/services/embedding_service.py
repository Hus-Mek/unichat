"""Singleton embedding model loader to avoid reloading on every request."""

from sentence_transformers import SentenceTransformer

from ..core.config import get_settings

_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    """Return the shared embedding model, loading it on first call."""
    global _model
    if _model is None:
        settings = get_settings()
        _model = SentenceTransformer(settings.EMBED_MODEL)
    return _model
