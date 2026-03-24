"""
University RAG Assistant
Professional RAG system with 3-level access control
"""

from .config import Config
from .document_processor import DocumentProcessor
from .access_control import AccessController
from .llm_client import LLMClient
from .rag_engine import RAGEngine

__version__ = "1.0.0"
__all__ = [
    "Config",
    "DocumentProcessor",
    "AccessController",
    "LLMClient",
    "RAGEngine"
]