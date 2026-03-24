"""
Centralized configuration using Pydantic Settings.
All values are driven by environment variables with sensible defaults.
"""

from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache

from pydantic_settings import BaseSettings


class AccessLevel(IntEnum):
    """Five-level hierarchical access control."""
    PUBLIC = 0
    STUDENT = 1
    FACULTY = 2
    ADMIN_STAFF = 3
    EXECUTIVE_BOARD = 4


ACCESS_HIERARCHY: dict[AccessLevel, list[AccessLevel]] = {
    AccessLevel.PUBLIC: [AccessLevel.PUBLIC],
    AccessLevel.STUDENT: [AccessLevel.PUBLIC, AccessLevel.STUDENT],
    AccessLevel.FACULTY: [AccessLevel.PUBLIC, AccessLevel.STUDENT, AccessLevel.FACULTY],
    AccessLevel.ADMIN_STAFF: [
        AccessLevel.PUBLIC, AccessLevel.STUDENT,
        AccessLevel.FACULTY, AccessLevel.ADMIN_STAFF,
    ],
    AccessLevel.EXECUTIVE_BOARD: [
        AccessLevel.PUBLIC, AccessLevel.STUDENT,
        AccessLevel.FACULTY, AccessLevel.ADMIN_STAFF,
        AccessLevel.EXECUTIVE_BOARD,
    ],
}


@dataclass(frozen=True)
class ModelConfig:
    """LLM model metadata."""
    provider_id: str
    description: str
    input_cost: float   # USD per 1M tokens
    output_cost: float  # USD per 1M tokens


MODELS: dict[str, ModelConfig] = {
    "llama-4-scout-17b-16e-instruct": ModelConfig(
        provider_id="meta-llama/llama-4-scout-17b-16e-instruct",
        description="Llama 4 Scout | Fast & efficient",
        input_cost=0.10,
        output_cost=0.10,
    ),
    "llama-3.3-70b-versatile": ModelConfig(
        provider_id="llama-3.3-70b-versatile",
        description="Llama 3.3 70B | Best quality",
        input_cost=0.59,
        output_cost=0.79,
    ),
    "qwen-3-32b": ModelConfig(
        provider_id="qwen-3-32b",
        description="Qwen 3 32B | Advanced reasoning",
        input_cost=0.27,
        output_cost=0.27,
    ),
    "gpt-oss-120b": ModelConfig(
        provider_id="gpt-oss-120b",
        description="GPT OSS 120B | Premium quality",
        input_cost=0.60,
        output_cost=0.60,
    ),
    "gpt-oss-20b": ModelConfig(
        provider_id="gpt-oss-20b",
        description="GPT OSS 20B | Balanced",
        input_cost=0.10,
        output_cost=0.10,
    ),
    "kimi-k2": ModelConfig(
        provider_id="kimi-k2",
        description="Kimi K2 | Long context",
        input_cost=0.15,
        output_cost=0.15,
    ),
}


class Settings(BaseSettings):
    """Application settings, loaded from environment / .env file."""

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/uni_chatbot"

    # JWT
    SECRET_KEY: str = "change-me-to-a-strong-random-secret"
    ACCESS_TOKEN_EXPIRE_HOURS: int = 8

    # LLM provider selection
    LLM_PROVIDER: str = "groq"  # groq | vllm | ollama | azure
    GROQ_API_KEY: str = ""
    VLLM_BASE_URL: str = "http://localhost:8000/v1"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_KEY: str = ""
    AZURE_OPENAI_DEPLOYMENT: str = ""

    # Embedding
    EMBED_MODEL: str = "all-MiniLM-L6-v2"

    # ChromaDB
    CHROMA_DIR: str = "./chroma_data"

    # RAG
    CHUNK_SIZE: int = 1200
    CHUNK_OVERLAP: int = 200
    DEFAULT_N_RESULTS: int = 15
    DEFAULT_MODEL: str = "llama-3.3-70b-versatile"

    # Rate limiting
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 30
    MIN_REQUEST_INTERVAL_SECONDS: float = 1.0

    model_config = {"env_file": ".env", "extra": "ignore"}


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()


def get_model_config(model_key: str) -> ModelConfig:
    """Resolve a model key to its configuration, with fallback."""
    if model_key in MODELS:
        return MODELS[model_key]
    # Fallback chain
    for candidate in ("llama-3.3-70b-versatile", "llama-4-scout-17b-16e-instruct"):
        if candidate in MODELS:
            return MODELS[candidate]
    return next(iter(MODELS.values()))
