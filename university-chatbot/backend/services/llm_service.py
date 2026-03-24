"""LLM provider abstraction supporting Groq, vLLM, and Ollama."""

from abc import ABC, abstractmethod

from ..core.config import get_model_config, get_settings
from ..rag.prompt_templates import SYSTEM_PROMPT, build_user_prompt


class LLMProvider(ABC):
    """Base class for LLM providers."""

    @abstractmethod
    def generate(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> dict:
        ...


class GroqProvider(LLMProvider):
    """Groq API provider."""

    def __init__(self, api_key: str):
        from groq import Groq
        self._client = Groq(api_key=api_key)

    def generate(self, messages, model, max_tokens=2048, temperature=0.1):
        try:
            model_cfg = get_model_config(model)
            completion = self._client.chat.completions.create(
                model=model_cfg.provider_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            usage = completion.usage
            return {
                "text": completion.choices[0].message.content,
                "tokens": {
                    "prompt": usage.prompt_tokens,
                    "completion": usage.completion_tokens,
                    "total": usage.total_tokens,
                },
                "finish_reason": completion.choices[0].finish_reason,
                "success": True,
            }
        except Exception as e:
            error_msg = str(e)
            is_rate_limit = "rate_limit" in error_msg.lower()
            return {
                "text": "Rate limit reached. Please wait." if is_rate_limit else f"Error: {error_msg}",
                "tokens": None,
                "finish_reason": "error",
                "success": False,
                "error": "rate_limit" if is_rate_limit else error_msg,
            }


class OpenAICompatibleProvider(LLMProvider):
    """Provider for vLLM, Ollama, or any OpenAI-compatible API."""

    def __init__(self, base_url: str, api_key: str = "not-needed"):
        from openai import OpenAI
        self._client = OpenAI(base_url=base_url, api_key=api_key)

    def generate(self, messages, model, max_tokens=2048, temperature=0.1):
        try:
            completion = self._client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            usage = completion.usage
            tokens = None
            if usage:
                tokens = {
                    "prompt": usage.prompt_tokens or 0,
                    "completion": usage.completion_tokens or 0,
                    "total": usage.total_tokens or 0,
                }
            return {
                "text": completion.choices[0].message.content,
                "tokens": tokens,
                "finish_reason": completion.choices[0].finish_reason,
                "success": True,
            }
        except Exception as e:
            return {
                "text": f"Error: {e}",
                "tokens": None,
                "finish_reason": "error",
                "success": False,
                "error": str(e),
            }


class LLMService:
    """Factory that selects and wraps the configured LLM provider."""

    def __init__(self):
        settings = get_settings()
        self.provider = self._create_provider(settings)
        self._settings = settings

    def _create_provider(self, settings) -> LLMProvider:
        if settings.LLM_PROVIDER == "groq":
            return GroqProvider(settings.GROQ_API_KEY)
        elif settings.LLM_PROVIDER == "vllm":
            return OpenAICompatibleProvider(settings.VLLM_BASE_URL)
        elif settings.LLM_PROVIDER == "ollama":
            return OpenAICompatibleProvider(f"{settings.OLLAMA_BASE_URL}/v1")
        elif settings.LLM_PROVIDER == "azure":
            return OpenAICompatibleProvider(
                base_url=settings.AZURE_OPENAI_ENDPOINT,
                api_key=settings.AZURE_OPENAI_KEY,
            )
        else:
            raise ValueError(f"Unknown LLM provider: {settings.LLM_PROVIDER}")

    def query(
        self,
        question: str,
        context: str,
        sources: list[str],
        model: str | None = None,
        max_tokens: int = 2048,
    ) -> dict:
        """Build the RAG prompt and query the LLM."""
        model = model or self._settings.DEFAULT_MODEL
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(question, context, sources)},
        ]
        return self.provider.generate(messages, model, max_tokens)

    @staticmethod
    def calculate_cost(tokens: dict | None, model: str) -> dict:
        """Calculate the USD cost of a query from token usage."""
        if not tokens:
            return {"input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}
        cfg = get_model_config(model)
        input_cost = (tokens["prompt"] / 1_000_000) * cfg.input_cost
        output_cost = (tokens["completion"] / 1_000_000) * cfg.output_cost
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
        }
