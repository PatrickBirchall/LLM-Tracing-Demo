from functools import lru_cache
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field, HttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


load_dotenv()


class AppSettings(BaseSettings):
    """Application configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_base_url: HttpUrl = Field(
        "https://openrouter.ai/api/v1",
        env="OPENAI_BASE_URL",
        description="Base URL for OpenRouter-compatible OpenAI client.",
    )

    langfuse_public_key: Optional[str] = Field(
        default=None,
        env="LANGFUSE_PUBLIC_KEY",
        description="Langfuse public API key.",
    )
    langfuse_secret_key: Optional[str] = Field(
        default=None,
        env="LANGFUSE_SECRET_KEY",
        description="Langfuse secret API key.",
    )
    langfuse_host: HttpUrl = Field(
        "http://localhost:3000",
        env="LANGFUSE_HOST",
        description="Langfuse host URL.",
    )


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """
    Return cached application settings.

    This validates that required environment variables are present at startup.
    """
    return AppSettings()

