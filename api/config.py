"""
Configuration management for the API
"""

from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import Field


class APIConfig(BaseSettings):
    """API configuration using Pydantic BaseSettings for environment variable handling"""

    # Server configuration
    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=False, env="API_DEBUG")

    # CORS configuration
    cors_origins: list[str] = Field(
        default=["http://localhost:3000"], env="CORS_ORIGINS"
    )
    cors_allow_credentials: bool = Field(default=True, env="CORS_ALLOW_CREDENTIALS")

    # RAG system paths
    index_path: Path = Field(default=Path("./index.db"), env="INDEX_PATH")
    templates_path: Path = Field(default=Path("templates"), env="TEMPLATES_PATH")

    # API limits
    max_question_length: int = Field(default=1000, env="MAX_QUESTION_LENGTH")

    # Logging
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    model_config = {"env_file": ".env", "case_sensitive": False}


def get_config() -> APIConfig:
    """Get API configuration instance"""
    return APIConfig()
