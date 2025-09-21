"""
Unit tests for logging configuration
"""

import pytest
import logging
from app.ai_engine.utils.logging_config import (
    get_logger,
    get_ai_logger,
    AIEngineLogger,
    setup_logging
)


class TestLoggingConfig:
    """Test logging configuration and functions"""

    def test_get_logger_returns_ai_engine_logger(self):
        """Test that get_logger returns an AIEngineLogger instance"""
        logger = get_logger("test_module")
        assert isinstance(logger, AIEngineLogger)
        assert logger.component_name == "test_module"

    def test_get_ai_logger_returns_same_type(self):
        """Test that get_ai_logger returns the same type as get_logger"""
        logger1 = get_logger("test1")
        logger2 = get_ai_logger("test2")
        assert type(logger1) == type(logger2)

    def test_logger_has_correct_methods(self):
        """Test that AIEngineLogger has all expected methods"""
        logger = get_logger("test")

        # Check that all logging methods exist
        assert hasattr(logger, 'log_processing_start')
        assert hasattr(logger, 'log_processing_end')
        assert hasattr(logger, 'log_performance_metric')
        assert hasattr(logger, 'log_detection_result')
        assert hasattr(logger, 'log_stock_analysis')
        assert hasattr(logger, 'log_error_with_context')

    def test_setup_logging_creates_handlers(self, tmp_path):
        """Test that setup_logging creates appropriate handlers"""
        log_file = tmp_path / "test.log"

        setup_logging(
            log_level="INFO",
            log_file=str(log_file),
            max_file_size=1024,
            backup_count=2
        )

        root_logger = logging.getLogger()

        # Check that handlers were created
        assert len(root_logger.handlers) > 0

        # Check log level
        assert root_logger.level == logging.INFO

    def test_logger_log_processing_start(self, caplog):
        """Test log_processing_start method"""
        logger = get_logger("test_module")

        with caplog.at_level(logging.INFO):
            logger.log_processing_start(
                "test_operation",
                {"param1": "value1", "param2": 42}
            )

        assert "Starting test_operation" in caplog.text
        assert "param1=value1" in caplog.text
        assert "param2=42" in caplog.text

    def test_logger_log_processing_end_success(self, caplog):
        """Test log_processing_end method for successful operations"""
        logger = get_logger("test_module")

        with caplog.at_level(logging.INFO):
            logger.log_processing_end(
                "test_operation",
                success=True,
                duration=1.5,
                details={"result": "success"}
            )

        assert "test_operation completed in 1.50s" in caplog.text
        assert "result=success" in caplog.text

    def test_logger_log_processing_end_failure(self, caplog):
        """Test log_processing_end method for failed operations"""
        logger = get_logger("test_module")

        with caplog.at_level(logging.ERROR):
            logger.log_processing_end(
                "test_operation",
                success=False,
                duration=0.5,
                details={"error": "test error"}
            )

        assert "test_operation failed in 0.50s" in caplog.text
        assert "error=test error" in caplog.text

    def test_logger_log_performance_metric(self, caplog):
        """Test log_performance_metric method"""
        logger = get_logger("test_module")

        with caplog.at_level(logging.INFO):
            logger.log_performance_metric("processing_time", 2.5, "seconds")

        assert "Performance metric: processing_time = 2.50 seconds" in caplog.text

    def test_logger_log_detection_result(self, caplog):
        """Test log_detection_result method"""
        logger = get_logger("test_module")

        with caplog.at_level(logging.INFO):
            logger.log_detection_result("Product A", 5, 0.95)

        assert "Product detection: Product A" in caplog.text
        assert "5 objects detected" in caplog.text
        assert "avg confidence: 0.95" in caplog.text

    def test_logger_log_stock_analysis(self, caplog):
        """Test log_stock_analysis method"""
        logger = get_logger("test_module")

        with caplog.at_level(logging.INFO):
            logger.log_stock_analysis("Product B", 75.5, "high", 0.88)

        assert "Stock analysis: Product B" in caplog.text
        assert "75.5% full" in caplog.text
        assert "high" in caplog.text
        assert "confidence: 0.88" in caplog.text

    def test_logger_log_error_with_context(self, caplog):
        """Test log_error_with_context method"""
        logger = get_logger("test_module")

        test_error = ValueError("Test error message")
        context = {"operation": "test", "user_id": 123}

        with caplog.at_level(logging.ERROR):
            logger.log_error_with_context(test_error, context)

        assert "Error in test_module" in caplog.text
        assert "Test error message" in caplog.text
        assert "operation=test" in caplog.text
        assert "user_id=123" in caplog.text

    def test_backward_compatibility(self):
        """Test that get_logger provides backward compatibility"""
        # This should not raise any errors
        logger = get_logger("backward_compat_test")

        # Should work exactly like get_ai_logger
        ai_logger = get_ai_logger("backward_compat_test")

        assert type(logger) == type(ai_logger)
        assert logger.component_name == ai_logger.component_name