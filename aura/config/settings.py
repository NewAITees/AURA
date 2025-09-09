"""
AURA Configuration Settings
"""

import os
from pathlib import Path
from typing import List, Optional

from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings."""
    
    # Application
    debug: bool = Field(False, env="DEBUG")
    development_mode: bool = Field(False, env="DEVELOPMENT_MODE")
    secret_key: str = Field("dev-secret-key", env="SECRET_KEY")
    
    # Redis Configuration
    redis_url: str = Field("redis://localhost:6379", env="REDIS_URL")
    redis_password: Optional[str] = Field(None, env="REDIS_PASSWORD")
    
    # Ollama Configuration
    ollama_url: str = Field("http://localhost:11434", env="OLLAMA_URL")
    ollama_models: List[str] = Field(
        ["llama3.1:8b", "mixtral:8x7b"], 
        env="OLLAMA_MODELS"
    )
    
    # Database Configuration
    database_url: str = Field("sqlite:///aura.db", env="DATABASE_URL")
    
    # Docker Configuration
    docker_host: str = Field("unix:///var/run/docker.sock", env="DOCKER_HOST")
    
    # Logging Configuration
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_file: str = Field("logs/aura.log", env="LOG_FILE")
    
    # TUI Configuration
    tui_host: str = Field("0.0.0.0", env="TUI_HOST")
    tui_port: int = Field(8000, env="TUI_PORT")
    
    # Agent Configuration
    max_concurrent_agents: int = Field(5, env="MAX_CONCURRENT_AGENTS")
    agent_timeout: int = Field(300, env="AGENT_TIMEOUT")
    
    # Model Configuration
    default_model: str = Field("llama3.1:8b", env="DEFAULT_MODEL")
    fallback_model: str = Field("llama3.1:8b", env="FALLBACK_MODEL")
    
    # Execution Environment Configuration
    execution_timeout: int = Field(300, env="EXECUTION_TIMEOUT")
    memory_limit: str = Field("512m", env="MEMORY_LIMIT")
    cpu_limit: float = Field(1.0, env="CPU_LIMIT")
    
    # Monitoring Configuration
    enable_metrics: bool = Field(True, env="ENABLE_METRICS")
    metrics_port: int = Field(9090, env="METRICS_PORT")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Global settings instance
settings = Settings()


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


def get_logs_dir() -> Path:
    """Get the logs directory, creating if it doesn't exist."""
    logs_dir = get_project_root() / "logs"
    logs_dir.mkdir(exist_ok=True)
    return logs_dir


def get_cache_dir() -> Path:
    """Get the cache directory, creating if it doesn't exist."""
    cache_dir = get_project_root() / "cache"
    cache_dir.mkdir(exist_ok=True)
    return cache_dir


def get_data_dir() -> Path:
    """Get the data directory, creating if it doesn't exist."""
    data_dir = get_project_root() / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir