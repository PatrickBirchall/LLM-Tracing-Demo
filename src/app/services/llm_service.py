from typing import Optional

from openai import OpenAI
from langfuse import observe, propagate_attributes

from app.config import get_settings


class LLMServiceError(Exception):
    """Raised when the LLM/OpenRouter API call fails."""


def _get_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(
        base_url=str(settings.openai_base_url),
        api_key=settings.openai_api_key,
    )


@observe(name="llm-chat-completion", as_type="generation")
def get_llm_response(prompt: str, request_id: Optional[str] = None) -> str:
    """
    Call the LLM via OpenRouter and return the assistant's response content.

    This function is traced by Langfuse via the @observe decorator.
    When a FastAPI request ID is provided, it is attached as metadata on the trace.
    """
    client = _get_client()

    def _call_llm() -> str:
        try:
            completion = client.chat.completions.create(
                model="anthropic/claude-3.5-sonnet",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant for a tracing demo.",
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMServiceError("Failed to call LLM via OpenRouter") from exc

        try:
            message = completion.choices[0].message
            content = getattr(message, "content", None)
            if isinstance(content, str):
                if content.strip():
                    return content
                raise LLMServiceError("Empty response from LLM")
            # OpenAI v1 may return a list of content parts
            if isinstance(content, list) and content:
                # Join text segments if present
                text_parts = [
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in content
                ]
                combined = "\n".join(t for t in text_parts if t)
                if combined.strip():
                    return combined
        except Exception as exc:  # noqa: BLE001
            raise LLMServiceError("Unexpected LLM response structure") from exc

        raise LLMServiceError("Empty response from LLM")

    if request_id:
        # Attach FastAPI request ID to the current Langfuse trace metadata
        with propagate_attributes(metadata={"request_id": request_id}):
            return _call_llm()

    return _call_llm()

