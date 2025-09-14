import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    max_file_size: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    format_string: Optional[str] = None
):
    """
    Set up comprehensive logging configuration for the AI engine
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        max_file_size: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        format_string: Custom format string for log messages
    """
    
    # Default format string
    if format_string is None:
        format_string = (
            '%(asctime)s - %(name)s - %(levelname)s - '
            '[%(filename)s:%(lineno)d] - %(message)s'
        )
    
    # Create formatter
    formatter = logging.Formatter(
        format_string,
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Set up root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(getattr(logging, log_level.upper()))
    root_logger.addHandler(console_handler)
    
    # File handler (if log_file is specified)
    if log_file:
        try:
            # Create log directory if it doesn't exist
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir, exist_ok=True)
            
            # Rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_file_size,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setFormatter(formatter)
            file_handler.setLevel(getattr(logging, log_level.upper()))
            root_logger.addHandler(file_handler)
            
        except Exception as e:
            print(f"Warning: Failed to set up file logging: {e}")
    
    # Set specific logger levels for external libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)
    
    # AI Engine specific loggers
    ai_loggers = [
        "models.model_loader",
        "preprocessing.pipeline", 
        "inference.engine",
        "inference.product_detection",
        "inference.stock_calculator",
        "api.main"
    ]
    
    for logger_name in ai_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(getattr(logging, log_level.upper()))

class AIEngineLogger:
    """Custom logger class for AI engine components"""
    
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.component_name = name
    
    def log_processing_start(self, operation: str, details: Optional[dict] = None):
        """Log the start of a processing operation"""
        message = f"Starting {operation}"
        if details:
            detail_str = ", ".join([f"{k}={v}" for k, v in details.items()])
            message += f" - {detail_str}"
        self.logger.info(message)
    
    def log_processing_end(self, operation: str, success: bool, 
                          duration: float, details: Optional[dict] = None):
        """Log the end of a processing operation"""
        status = "completed" if success else "failed"
        message = f"{operation} {status} in {duration:.2f}s"
        
        if details:
            detail_str = ", ".join([f"{k}={v}" for k, v in details.items()])
            message += f" - {detail_str}"
        
        if success:
            self.logger.info(message)
        else:
            self.logger.error(message)
    
    def log_performance_metric(self, metric_name: str, value: float, unit: str = ""):
        """Log performance metrics"""
        message = f"Performance metric: {metric_name} = {value:.2f}"
        if unit:
            message += f" {unit}"
        self.logger.info(message)
    
    def log_detection_result(self, product: str, detections: int, confidence: float):
        """Log product detection results"""
        self.logger.info(
            f"Product detection: {product} - {detections} objects detected, "
            f"avg confidence: {confidence:.2f}"
        )
    
    def log_stock_analysis(self, product: str, percentage: float, 
                          abundance: str, confidence: float):
        """Log stock analysis results"""
        self.logger.info(
            f"Stock analysis: {product} - {percentage:.1f}% full, "
            f"{abundance}, confidence: {confidence:.2f}"
        )
    
    def log_error_with_context(self, error: Exception, context: dict):
        """Log error with additional context"""
        context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
        self.logger.error(f"Error in {self.component_name}: {str(error)} - Context: {context_str}")
        self.logger.debug("Full traceback:", exc_info=True)

def get_ai_logger(name: str) -> AIEngineLogger:
    """Get an AI engine logger instance"""
    return AIEngineLogger(name)

# Performance logging decorator
def log_performance(logger: AIEngineLogger, operation_name: str):
    """Decorator to log performance of functions"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            
            try:
                logger.log_processing_start(operation_name, {"function": func.__name__})
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                logger.log_processing_end(operation_name, True, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.log_processing_end(operation_name, False, duration)
                logger.log_error_with_context(e, {"function": func.__name__})
                raise
        
        def sync_wrapper(*args, **kwargs):
            import time
            start_time = time.time()
            
            try:
                logger.log_processing_start(operation_name, {"function": func.__name__})
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                logger.log_processing_end(operation_name, True, duration)
                return result
            except Exception as e:
                duration = time.time() - start_time
                logger.log_processing_end(operation_name, False, duration)
                logger.log_error_with_context(e, {"function": func.__name__})
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator
