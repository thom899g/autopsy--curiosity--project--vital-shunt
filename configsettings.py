"""
VITAL SHUNT Configuration Management
Centralized settings with validation and environment awareness
"""
from pydantic import Field
from pydantic_settings import BaseSettings
from typing import Optional, List
from enum import Enum


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class ModelProvider(str, Enum):
    DEEPSEEK = "deepseek"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


class Settings(BaseSettings):
    # Environment
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    debug: bool = Field(default=False)
    
    # API Configuration
    deepseek_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    
    # Firebase
    firebase_project_id: str
    firebase_credentials_path: str = "./config/firebase-credentials.json"
    
    # Model Settings
    default_model: ModelProvider = ModelProvider.DEEPSEEK
    fallback_sequence: List[ModelProvider] = [
        ModelProvider.DEEPSEEK,
        ModelProvider.OPENAI,
        ModelProvider.ANTHROPIC
    ]
    
    # Timeouts & Limits
    request_timeout_seconds: int = 30
    max_retries: int = 3
    circuit_breaker_threshold: int = 5
    circuit_breaker_reset_seconds: int = 60
    
    # Monitoring
    enable_telemetry: bool = True
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()