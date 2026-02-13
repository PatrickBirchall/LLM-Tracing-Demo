from typing import Optional

from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from .config import get_settings
from .services.llm_service import LLMServiceError, get_llm_response
from langfuse import get_client


settings = get_settings()
langfuse_client = get_client()

app = FastAPI(
    title="LLM Tracing Demo",
    version="0.1.0",
    description="FastAPI application demonstrating LLM tracing with OpenRouter and Langfuse.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    """
    Ensure every request has a unique request_id available on request.state.
    Prefer X-Request-ID header if provided, otherwise generate a UUID4.
    """
    header_request_id = request.headers.get("X-Request-ID")
    request_id = header_request_id or str(uuid4())
    request.state.request_id = request_id

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


class ChatRequest(BaseModel):
    message: str
    model: Optional[str] = None
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    response: str
    request_id: str


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(payload: ChatRequest, request: Request) -> ChatResponse:
    """
    Simple chat endpoint that delegates to the LLM service.

    The FastAPI request ID and session ID are propagated into the Langfuse trace
    via the service layer.
    """
    request_id: str = getattr(request.state, "request_id", str(uuid4()))
    session_id: Optional[str] = payload.session_id or request.headers.get("X-Session-ID")

    # Run blocking LLM call in a threadpool so we don't block the event loop.
    llm_response: str = await run_in_threadpool(
        get_llm_response,
        payload.message,
        request_id,
        session_id,
        payload.model,
    )

    return ChatResponse(response=llm_response, request_id=request_id)


def _record_error_span(
    request: Request,
    error: Exception,
    error_type: str,
) -> None:
    """
    Create a Langfuse span recording an error before returning a 500 response.

    This ensures errors are visible in Langfuse even if they occur outside
    of an observed function.
    """
    if langfuse_client is None:
        return

    request_id: Optional[str] = getattr(request.state, "request_id", None)

    with langfuse_client.start_as_current_span(
        name=error_type,
        input={
            "path": str(request.url.path),
            "method": request.method,
            "request_id": request_id,
        },
        metadata={
            "error": str(error),
            "error_type": error_type,
        },
        level="error",
        status_message="error",
    ):
        # Nothing else to do in the span body; span context captures the error.
        ...


@app.exception_handler(LLMServiceError)
async def llm_service_exception_handler(
    request: Request,
    exc: LLMServiceError,
) -> JSONResponse:
    """
    Handle errors originating from the LLM service and report them to Langfuse.
    """
    _record_error_span(request, exc, error_type="LLMServiceError")

    request_id: Optional[str] = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "An error occurred while calling the LLM.",
            "request_id": request_id,
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """
    Catch-all handler for unexpected errors and report them to Langfuse.
    """
    _record_error_span(request, exc, error_type="UnhandledException")

    request_id: Optional[str] = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error.",
            "request_id": request_id,
        },
    )

