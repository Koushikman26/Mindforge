"""
Configuration management using Pydantic Settings

This module provides centralized configuration management for the entire application
using environment variables and Pydantic for validation.
"""

from functools import lru_cache
from typing import List, Optional
from pydantic_settings import BaseSettings
from pydantic import Field, validator
import os


class Settings(BaseSettings):
    """Application settings with environment variable support"""

    # Application Info
    app_name: str = Field(default="Mindforge", env="APP_NAME")
    app_version: str = Field(default="1.1.0", env="APP_VERSION")
    environment: str = Field(default="development", env="ENVIRONMENT")
    debug: bool = Field(default=False, env="DEBUG")

    # Server Configuration
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    reload: bool = Field(default=False, env="RELOAD")

    # API Configuration
    api_v1_prefix: str = Field(default="/api/v1", env="API_V1_PREFIX")
    openapi_url: Optional[str] = Field(default="/openapi.json", env="OPENAPI_URL")
    docs_url: Optional[str] = Field(default="/docs", env="DOCS_URL")
    redoc_url: Optional[str] = Field(default="/redoc", env="REDOC_URL")

    # CORS Configuration
    allowed_origins: List[str] = Field(
        default=["*"],
        env="ALLOWED_ORIGINS"
    )
    allowed_methods: List[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        env="ALLOWED_METHODS"
    )
    allowed_headers: List[str] = Field(
        default=["*"],
        env="ALLOWED_HEADERS"
    )
    allow_credentials: bool = Field(default=True, env="ALLOW_CREDENTIALS")

    # AI Engine Configuration
    ai_engine_url: str = Field(
        default="http://ai-engine:8001",
        env="AI_ENGINE_URL"
    )
    ai_engine_timeout: int = Field(default=60, env="AI_ENGINE_TIMEOUT")
    ai_max_retries: int = Field(default=3, env="AI_MAX_RETRIES")

    # Image Processing Configuration
    max_image_size_mb: int = Field(default=10, env="MAX_IMAGE_SIZE_MB")
    max_batch_size: int = Field(default=10, env="MAX_BATCH_SIZE")
    image_quality: int = Field(default=95, env="IMAGE_QUALITY")
    processing_timeout: int = Field(default=30, env="PROCESSING_TIMEOUT")
    enable_gpu: bool = Field(default=False, env="ENABLE_GPU")

    # Performance Settings
    max_concurrent_processing: int = Field(default=2, env="MAX_CONCURRENT_PROCESSING")
    memory_limit_mb: int = Field(default=2048, env="MEMORY_LIMIT_MB")
    cache_enabled: bool = Field(default=True, env="CACHE_ENABLED")
    cache_ttl: int = Field(default=3600, env="CACHE_TTL")

    # Logging Configuration
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_file: Optional[str] = Field(default=None, env="LOG_FILE")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    log_max_size_mb: int = Field(default=10, env="LOG_MAX_SIZE_MB")
    log_backup_count: int = Field(default=5, env="LOG_BACKUP_COUNT")

    # Security Configuration
    secret_key: str = Field(
        default="change-this-secret-key-in-production",
        env="SECRET_KEY"
    )
    algorithm: str = Field(default="HS256", env="ALGORITHM")
    access_token_expire_minutes: int = Field(default=30, env="ACCESS_TOKEN_EXPIRE_MINUTES")

    # Database Configuration (for future use)
    database_url: Optional[str] = Field(default=None, env="DATABASE_URL")
    database_pool_size: int = Field(default=5, env="DATABASE_POOL_SIZE")
    database_max_overflow: int = Field(default=10, env="DATABASE_MAX_OVERFLOW")

    # Redis Configuration (for caching)
    redis_url: Optional[str] = Field(default=None, env="REDIS_URL")
    redis_password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    redis_db: int = Field(default=0, env="REDIS_DB")

    # File Storage
    upload_dir: str = Field(default="/tmp/uploads", env="UPLOAD_DIR")
    temp_dir: str = Field(default="/tmp/temp", env="TEMP_DIR")
    max_upload_size_mb: int = Field(default=50, env="MAX_UPLOAD_SIZE_MB")

    @validator("allowed_origins", pre=True)
    def parse_cors_origins(cls, v):
        """Parse comma-separated CORS origins"""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",")]
        return v

    @validator("environment")
    def validate_environment(cls, v):
        """Validate environment value"""
        allowed = ["development", "staging", "production", "testing"]
        if v not in allowed:
            raise ValueError(f"Environment must be one of: {', '.join(allowed)}")
        return v

    @validator("log_level")
    def validate_log_level(cls, v):
        """Validate log level"""
        allowed = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        v = v.upper()
        if v not in allowed:
            raise ValueError(f"Log level must be one of: {', '.join(allowed)}")
        return v

    @property
    def max_image_size_bytes(self) -> int:
        """Calculate max image size in bytes"""
        return self.max_image_size_mb * 1024 * 1024

    @property
    def max_upload_size_bytes(self) -> int:
        """Calculate max upload size in bytes"""
        return self.max_upload_size_mb * 1024 * 1024

    @property
    def is_production(self) -> bool:
        """Check if running in production"""
        return self.environment == "production"

    @property
    def is_development(self) -> bool:
        """Check if running in development"""
        return self.environment == "development"

    class Config:
        """Pydantic configuration"""
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # Ignore extra fields in .env file


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance

    Returns:
        Settings: Application settings singleton
    """
    return Settings()


# Create a global settings instance
settings = get_settings()


# Helper functions for common settings access
def get_log_config() -> dict:
    """Get logging configuration dict"""
    return {
        "level": settings.log_level,
        "format": settings.log_format,
        "file": settings.log_file,
        "max_size": settings.log_max_size_mb * 1024 * 1024,
        "backup_count": settings.log_backup_count,
    }


def get_cors_config() -> dict:
    """Get CORS configuration dict"""
    return {
        "allow_origins": settings.allowed_origins,
        "allow_methods": settings.allowed_methods,
        "allow_headers": settings.allowed_headers,
        "allow_credentials": settings.allow_credentials,
    }


def get_image_processing_config() -> dict:
    """Get image processing configuration dict"""
    return {
        "max_size_mb": settings.max_image_size_mb,
        "max_batch_size": settings.max_batch_size,
        "quality": settings.image_quality,
        "timeout": settings.processing_timeout,
        "enable_gpu": settings.enable_gpu,
        "max_concurrent": settings.max_concurrent_processing,
    }


# Environment-specific configuration overrides
if settings.is_production:
    # Production-specific settings
    settings.debug = False
    settings.reload = False
    if settings.secret_key == "change-this-secret-key-in-production":
        import warnings
        warnings.warn(
            "Using default secret key in production! Please set SECRET_KEY environment variable.",
            UserWarning
        )

elif settings.is_development:
    # Development-specific settings
    settings.debug = True
    settings.reload = True
    settings.log_level = "DEBUG"


# Print configuration summary on import (only in debug mode)
if settings.debug:
    print(f"[CONFIG] Mindforge Configuration Loaded")
    print(f"  Environment: {settings.environment}")
    print(f"  Debug Mode: {settings.debug}")
    print(f"  API URL: http://{settings.host}:{settings.port}")
    print(f"  AI Engine: {settings.ai_engine_url}")
    print(f"  Log Level: {settings.log_level}")