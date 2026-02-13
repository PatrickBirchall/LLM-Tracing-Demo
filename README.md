## LLM Tracing Demo

This project is a minimal FastAPI application that demonstrates LLM tracing using:

- FastAPI as the HTTP API layer
- OpenRouter as the LLM gateway (via the `openai` Python SDK)
- Langfuse for tracing using the `@observe()` decorator

The main endpoint is a `POST /chat` route that calls an LLM and links each FastAPI request to a Langfuse trace.

---

### Project layout

- `pyproject.toml` – uv/PEP 621 project configuration and dependencies
- `src/app/main.py` – FastAPI entrypoint and `/chat` route
- `src/app/config.py` – Pydantic settings loaded from environment / `.env`
- `src/app/services/llm_service.py` – Langfuse-observed LLM service logic
- `.env` – local environment variables (not committed)
- `.env.example` – example environment template checked into git

---

### Environment variables

Create a `.env` file (you can copy from `.env.example`) with at least:

- `OPENAI_API_KEY` – your OpenRouter API key
- `OPENAI_BASE_URL` – usually `https://openrouter.ai/api/v1`
- `LANGFUSE_PUBLIC_KEY` – Langfuse public key
- `LANGFUSE_SECRET_KEY` – Langfuse secret key
- `LANGFUSE_HOST` – Langfuse host, defaults to `http://localhost:3000`

The Langfuse SDK reads these environment variables automatically. This example assumes you are running Langfuse locally at `http://localhost:3000`.

---

### Setup with uv

From the project root:

1. **Create a virtual environment**

   ```bash
   uv venv
   ```

2. **Install dependencies**

   ```bash
   uv add fastapi uvicorn openai langfuse python-dotenv pydantic pydantic-settings
   ```

   (If you already have `pyproject.toml` checked in, this will resolve and install the versions into your uv-managed environment.)

3. **Run the FastAPI server**

   ```bash
   uv run uvicorn src.app.main:app --reload
   ```

The API will be available at `http://127.0.0.1:8000`.

---

### Example request

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Session-ID: demo-session-123" \
  -d '{"message": "Say hello and explain what tracing is."}'
```

You can optionally pass `model` and `session_id` in the body. The `session_id` can also be sent via the `X-Session-ID` header.

```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello!", "model": "openai/gpt-4o", "session_id": "demo-session-123"}'
```

Example JSON response:

```json
{
  "response": "…LLM-generated content…",
  "request_id": "c3f6d0a1-3f5a-4b4a-9e56-..."
}
```

You can use the `request_id` to correlate API calls with traces in Langfuse, where it is attached as metadata on the traced LLM call.
