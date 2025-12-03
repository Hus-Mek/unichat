"""
Configuration Module
Centralized configuration for the RAG system
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class ModelConfig:
    """LLM Model configuration"""
    name: str
    description: str
    input_cost: float  # per 1M tokens
    output_cost: float  # per 1M tokens


class Config:
    """Application configuration"""
    
    # Embedding model
    EMBED_MODEL = "all-MiniLM-L6-v2"
    
    # ChromaDB settings
    CHROMA_DIR = "demo_store"
    COLLECTION_NAME = "uni_docs"
    
    # Chunking settings
    CHUNK_SIZE = 1200
    CHUNK_OVERLAP = 200
    
    # Query settings
    DEFAULT_N_RESULTS = 15
    MIN_REQUEST_INTERVAL = 2  # seconds between requests
    
    # Access levels
    ACCESS_LEVELS = {
        "public": "Public",
        "student": "Student",
        "faculty": "Faculty"
    }
    
    # Access hierarchy (who can see what)
    ACCESS_HIERARCHY = {
        "public": ["public"],
        "student": ["public", "student"],
        "faculty": ["public", "student", "faculty"]
    }
    
    # Available LLM models
    MODELS: Dict[str, ModelConfig] = {
        "llama-3.3-70b-versatile": ModelConfig(
            name="llama-3.3-70b-versatile",
            description="70B | Best quality",
            input_cost=0.59,
            output_cost=0.79
        ),
        "llama-3.1-70b-versatile": ModelConfig(
            name="llama-3.1-70b-versatile",
            description="70B | Production",
            input_cost=0.59,
            output_cost=0.79
        ),
        "llama-3.1-8b-instant": ModelConfig(
            name="llama-3.1-8b-instant",
            description="8B | Fast",
            input_cost=0.05,
            output_cost=0.08
        ),
    }
    
    @classmethod
    def get_model_list(cls) -> List[str]:
        """Get list of available model names"""
        return list(cls.MODELS.keys())
    
    @classmethod
    def get_model_config(cls, model_name: str) -> ModelConfig:
        """Get configuration for a specific model"""
        return cls.MODELS.get(
            model_name, 
            cls.MODELS["llama-3.3-70b-versatile"]
        )