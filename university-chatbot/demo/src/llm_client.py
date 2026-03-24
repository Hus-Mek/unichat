"""
LLM Client Module
Handles interactions with Groq API
"""

from typing import Dict, Optional
from groq import Groq

from .config import Config


class LLMClient:
    """Handles LLM queries via Groq API"""
    
    def __init__(self, api_key: str):
        """
        Initialize LLM client
        
        Args:
            api_key: Groq API key
        """
        self.client = Groq(api_key=api_key)
        self.models = Config.MODELS
    
    def query(
        self,
        question: str,
        context: str,
        model: str = "llama-3.3-70b-versatile",
        max_tokens: int = 2048
    ) -> Dict:
        """
        Query the LLM with context
        
        Args:
            question: User's question
            context: Retrieved context from RAG
            model: Model name to use
            max_tokens: Maximum tokens in response
            
        Returns:
            Dictionary with response text, tokens, and metadata
        """
        prompt = self._build_prompt(question, context)
        
        try:
            # Resolve model ID from config (allows friendly keys vs provider IDs)
            resolved = Config.get_model_config(model).name if model in self.models else model

            completion = self.client.chat.completions.create(
                model=resolved,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=max_tokens
            )
            
            usage = completion.usage
            response_text = completion.choices[0].message.content
            finish_reason = completion.choices[0].finish_reason
            
            return {
                "text": response_text,
                "tokens": {
                    "prompt": usage.prompt_tokens,
                    "completion": usage.completion_tokens,
                    "total": usage.total_tokens
                },
                "finish_reason": finish_reason,
                "success": True
            }
            
        except Exception as e:
            error_msg = str(e)
            if "rate_limit" in error_msg.lower():
                return {
                    "text": "⚠️ Rate limit reached. Please wait a moment.",
                    "tokens": None,
                    "finish_reason": "error",
                    "success": False,
                    "error": "rate_limit"
                }
            else:
                return {
                    "text": f"⚠️ Error: {error_msg}",
                    "tokens": None,
                    "finish_reason": "error",
                    "success": False,
                    "error": error_msg
                }
    
    def _build_prompt(self, question: str, context: str) -> str:
        """
        Build the prompt for the LLM
        
        Args:
            question: User's question
            context: Retrieved context
            
        Returns:
            Formatted prompt string
        """
        return f"""You are a university assistant. 
Answer the question using ONLY the context below.
If the answer is not in the context, say you don't know.

IMPORTANT INSTRUCTIONS:
1. If the question asks for a list, provide ALL results found in the context.
2. Look carefully at ALL information - relationships work both ways.
3. Search through the ENTIRE context before concluding information is unavailable.

CONTEXT:
\"\"\"{context}\"\"\"

QUESTION:
{question}"""
    
    def calculate_cost(
        self, 
        tokens: Dict[str, int], 
        model: str
    ) -> Dict[str, float]:
        """
        Calculate cost of query
        
        Args:
            tokens: Token usage dictionary
            model: Model name used
            
        Returns:
            Dictionary with cost breakdown
        """
        if not tokens:
            return {
                "input_cost": 0.0,
                "output_cost": 0.0,
                "total_cost": 0.0
            }
        
        model_config = Config.get_model_config(model)
        
        input_cost = (tokens["prompt"] / 1_000_000) * model_config.input_cost
        output_cost = (tokens["completion"] / 1_000_000) * model_config.output_cost
        total_cost = input_cost + output_cost
        
        return {
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost
        }