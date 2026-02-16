from typing import Any, Dict, Optional

from openai import OpenAI
from langfuse import observe, propagate_attributes

from ..config import get_settings


class LLMServiceError(Exception):
    """Raised when the LLM/OpenRouter API call fails."""


def _get_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(
        base_url=str(settings.openai_base_url),
        api_key=settings.openai_api_key,
    )


def _build_user_content(prompt: str, document_context: Optional[str] = None) -> str:
    """Build the user message, optionally including document context for Q&A."""
    if document_context and document_context.strip():
        return (
            "Use the following document content to answer the question.\n\n"
            "--- Document ---\n"
            f"{document_context.strip()}\n"
            "--- End document ---\n\n"
            f"Question: {prompt}"
        )
    return prompt


@observe(name="llm-tracing-demo", as_type="generation")
def get_llm_response(
    prompt: str,
    request_id: Optional[str] = None,
    session_id: Optional[str] = None,
    model: Optional[str] = None,
    document_context: Optional[str] = None,
) -> str:
    """
    Call the LLM via OpenRouter and return the assistant's response content.

    This function is traced by Langfuse via the @observe decorator.
    When a FastAPI request ID is provided, it is attached as metadata on the trace.
    If document_context is provided, it is included so the model can answer questions about it.
    """
    client = _get_client()
    user_content = _build_user_content(prompt, document_context)

    def _call_llm() -> str:
        try:
            completion = client.chat.completions.create(
                model=model or "anthropic/claude-3.5-sonnet",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant for a tracing demo. When given document content, answer questions about it accurately and concisely.",
                    },
                    {
                        "role": "user",
                        "content": user_content,
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

    attributes: Dict[str, Any] = {}

    if session_id:
        attributes["session_id"] = session_id

    if request_id:
        attributes["metadata"] = {"request_id": request_id}

    if attributes:
        # Attach FastAPI request/session identifiers to the current Langfuse trace
        with propagate_attributes(**attributes):
            return _call_llm()

    return _call_llm()

