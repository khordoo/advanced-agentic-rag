from functools import lru_cache

from pydantic_settings import BaseSettings


class AppConfig(BaseSettings):
    # Public model identifier exposed by /v1/models and expected by /v1/chat/completions
    model_id: str = "advanced-rag-agent"
    api_port: int = 8000
    webui_port: int = 3000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"
        env_prefix = "RAG_"


@lru_cache
def get_settings():
    return AppConfig()
