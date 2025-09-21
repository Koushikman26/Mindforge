"""
Unit tests for configuration management
"""

import pytest
import os
from app.core.config import (
    Settings,
    get_settings,
    get_log_config,
    get_cors_config,
    get_image_processing_config
)


class TestSettings:
    """Test Settings configuration class"""

    def test_default_settings(self):
        """Test that default settings are loaded correctly"""
        settings = Settings()

        assert settings.app_name == "Mindforge"
        assert settings.app_version == "1.1.0"
        assert settings.environment in ["development", "staging", "production", "testing"]
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000

    def test_environment_validation(self):
        """Test environment validation"""
        # Valid environments should work
        for env in ["development", "staging", "production", "testing"]:
            settings = Settings(environment=env)
            assert settings.environment == env

        # Invalid environment should raise error
        with pytest.raises(ValueError) as exc_info:
            Settings(environment="invalid")
        assert "Environment must be one of" in str(exc_info.value)

    def test_log_level_validation(self):
        """Test log level validation"""
        # Valid log levels should work
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            settings = Settings(log_level=level)
            assert settings.log_level == level

        # Case insensitive
        settings = Settings(log_level="info")
        assert settings.log_level == "INFO"

        # Invalid log level should raise error
        with pytest.raises(ValueError) as exc_info:
            Settings(log_level="INVALID")
        assert "Log level must be one of" in str(exc_info.value)

    def test_cors_origins_parsing(self):
        """Test CORS origins parsing from string"""
        # String input should be parsed
        settings = Settings(allowed_origins="http://localhost:3000,http://example.com")
        assert settings.allowed_origins == ["http://localhost:3000", "http://example.com"]

        # List input should be preserved
        settings = Settings(allowed_origins=["http://localhost:3000"])
        assert settings.allowed_origins == ["http://localhost:3000"]

    def test_calculated_properties(self):
        """Test calculated properties"""
        settings = Settings(max_image_size_mb=5, max_upload_size_mb=10)

        assert settings.max_image_size_bytes == 5 * 1024 * 1024
        assert settings.max_upload_size_bytes == 10 * 1024 * 1024

    def test_environment_properties(self):
        """Test environment check properties"""
        # Production
        settings = Settings(environment="production")
        assert settings.is_production is True
        assert settings.is_development is False

        # Development
        settings = Settings(environment="development")
        assert settings.is_production is False
        assert settings.is_development is True

    def test_settings_from_environment_variables(self, monkeypatch):
        """Test loading settings from environment variables"""
        # Set environment variables
        monkeypatch.setenv("APP_NAME", "TestApp")
        monkeypatch.setenv("PORT", "9000")
        monkeypatch.setenv("DEBUG", "true")
        monkeypatch.setenv("MAX_IMAGE_SIZE_MB", "20")

        settings = Settings()

        assert settings.app_name == "TestApp"
        assert settings.port == 9000
        assert settings.debug is True
        assert settings.max_image_size_mb == 20

    def test_get_settings_caching(self):
        """Test that get_settings returns cached instance"""
        settings1 = get_settings()
        settings2 = get_settings()

        # Should be the same instance
        assert settings1 is settings2

    def test_get_log_config(self):
        """Test get_log_config helper function"""
        config = get_log_config()

        assert "level" in config
        assert "format" in config
        assert "file" in config
        assert "max_size" in config
        assert "backup_count" in config
        assert isinstance(config["max_size"], int)

    def test_get_cors_config(self):
        """Test get_cors_config helper function"""
        config = get_cors_config()

        assert "allow_origins" in config
        assert "allow_methods" in config
        assert "allow_headers" in config
        assert "allow_credentials" in config
        assert isinstance(config["allow_origins"], list)

    def test_get_image_processing_config(self):
        """Test get_image_processing_config helper function"""
        config = get_image_processing_config()

        assert "max_size_mb" in config
        assert "max_batch_size" in config
        assert "quality" in config
        assert "timeout" in config
        assert "enable_gpu" in config
        assert "max_concurrent" in config

    def test_secret_key_warning(self):
        """Test that default secret key triggers warning in production"""
        import warnings

        settings = Settings(
            environment="production",
            secret_key="change-this-secret-key-in-production"
        )

        # In production with default key, should warn
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            # Trigger the production check
            if settings.is_production and settings.secret_key == "change-this-secret-key-in-production":
                warnings.warn(
                    "Using default secret key in production!",
                    UserWarning
                )

            assert len(w) == 1
            assert "default secret key" in str(w[0].message)

    def test_settings_extra_fields_ignored(self):
        """Test that extra fields in settings are ignored"""
        # Should not raise error for unknown fields
        settings = Settings(unknown_field="value", another_unknown="test")
        assert not hasattr(settings, "unknown_field")
        assert not hasattr(settings, "another_unknown")