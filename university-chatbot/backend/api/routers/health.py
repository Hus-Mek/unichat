"""Health check endpoints."""

from fastapi import APIRouter

from ...core.config import MODELS, get_settings

router = APIRouter()


@router.get("/health")
async def health_check():
    """Basic health check."""
    return {"status": "healthy"}


@router.get("/health/models")
async def available_models():
    """List available LLM models."""
    return {
        "provider": get_settings().LLM_PROVIDER,
        "models": {
            key: {"description": m.description, "input_cost": m.input_cost, "output_cost": m.output_cost}
            for key, m in MODELS.items()
        },
    }
