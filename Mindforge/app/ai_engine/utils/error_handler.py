import logging
from typing import Dict, Any, Optional
from enum import Enum
import traceback
import time

logger = logging.getLogger(__name__)

class ErrorType(Enum):
    MODEL_LOADING_ERROR = "model_loading_error"
    PREPROCESSING_ERROR = "preprocessing_error"
    INFERENCE_ERROR = "inference_error"
    VALIDATION_ERROR = "validation_error"
    IO_ERROR = "io_error"
    CONFIGURATION_ERROR = "configuration_error"
    TIMEOUT_ERROR = "timeout_error"
    MEMORY_ERROR = "memory_error"

class AIErrorHandler:
    def __init__(self):
        self.error_counts = {}
        self.fallback_enabled = True
        self.max_retries = 3
        self.error_history = []
        self.max_history_size = 100
    
    def handle_error(self, error: Exception, error_type: ErrorType, 
                    context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Centralized error handling with detailed logging and fallback strategies"""
        error_info = {
            "error_type": error_type.value,
            "error_message": str(error),
            "traceback": traceback.format_exc(),
            "context": context or {},
            "timestamp": time.time()
        }
        
        # Log error with appropriate level
        if error_type in [ErrorType.MODEL_LOADING_ERROR, ErrorType.CONFIGURATION_ERROR]:
            logger.critical(f"{error_type.value}: {str(error)}")
        elif error_type in [ErrorType.INFERENCE_ERROR, ErrorType.IO_ERROR]:
            logger.error(f"{error_type.value}: {str(error)}")
        else:
            logger.warning(f"{error_type.value}: {str(error)}")
        
        # Update error counts and history
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        self._add_to_history(error_info)
        
        # Determine fallback strategy
        fallback_result = self._get_fallback_response(error_type, error_info)
        
        return {
            "success": False,
            "error_info": error_info,
            "fallback_result": fallback_result,
            "retry_recommended": self._should_retry(error_type),
            "retry_count": self._get_retry_count(error_type),
            "max_retries": self.max_retries
        }
    
    def _get_fallback_response(self, error_type: ErrorType, 
                              error_info: Dict[str, Any]) -> Dict[str, Any]:
        """Get fallback response based on error type"""
        fallback_responses = {
            ErrorType.MODEL_LOADING_ERROR: {
                "message": "AI model unavailable, using basic computer vision fallback",
                "fallback_available": True,
                "suggested_action": "Check internet connection and disk space",
                "estimated_accuracy": "60-70%"
            },
            ErrorType.PREPROCESSING_ERROR: {
                "message": "Image preprocessing failed, using original image",
                "fallback_available": True,
                "suggested_action": "Check image format and quality",
                "impact": "May reduce detection accuracy"
            },
            ErrorType.INFERENCE_ERROR: {
                "message": "AI inference failed, manual inspection recommended",
                "fallback_available": False,
                "suggested_action": "Retry with different image or check system resources",
                "impact": "No automated analysis available"
            },
            ErrorType.TIMEOUT_ERROR: {
                "message": "Processing timeout, try smaller image or retry",
                "fallback_available": True,
                "suggested_action": "Reduce image size or check system load",
                "impact": "Processing incomplete"
            },
            ErrorType.MEMORY_ERROR: {
                "message": "Insufficient memory, reduce image size",
                "fallback_available": True,
                "suggested_action": "Use smaller images or increase system memory",
                "impact": "May need image downscaling"
            },
            ErrorType.VALIDATION_ERROR: {
                "message": "Input validation failed, check data format",
                "fallback_available": False,
                "suggested_action": "Verify input parameters and data types",
                "impact": "Cannot process invalid input"
            },
            ErrorType.IO_ERROR: {
                "message": "File I/O error occurred",
                "fallback_available": True,
                "suggested_action": "Check file permissions and disk space",
                "impact": "May affect model loading or result saving"
            },
            ErrorType.CONFIGURATION_ERROR: {
                "message": "System configuration error",
                "fallback_available": False,
                "suggested_action": "Check configuration files and environment variables",
                "impact": "System may not function correctly"
            }
        }
        
        return fallback_responses.get(error_type, {
            "message": "Unknown error occurred",
            "fallback_available": False,
            "suggested_action": "Check system logs for details",
            "impact": "Unknown impact on functionality"
        })
    
    def _should_retry(self, error_type: ErrorType) -> bool:
        """Determine if operation should be retried"""
        retry_errors = {
            ErrorType.INFERENCE_ERROR,
            ErrorType.IO_ERROR,
            ErrorType.TIMEOUT_ERROR,
            ErrorType.PREPROCESSING_ERROR
        }
        
        no_retry_errors = {
            ErrorType.VALIDATION_ERROR,
            ErrorType.CONFIGURATION_ERROR
        }
        
        if error_type in no_retry_errors:
            return False
        
        if error_type in retry_errors:
            return self._get_retry_count(error_type) < self.max_retries
        
        return False
    
    def _get_retry_count(self, error_type: ErrorType) -> int:
        """Get current retry count for error type"""
        recent_errors = [
            e for e in self.error_history[-10:] 
            if e.get("error_type") == error_type.value and 
               time.time() - e.get("timestamp", 0) < 300  # Last 5 minutes
        ]
        return len(recent_errors)
    
    def _add_to_history(self, error_info: Dict[str, Any]):
        """Add error to history with size limit"""
        self.error_history.append(error_info)
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size:]
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics and health metrics"""
        try:
            total_errors = sum(self.error_counts.values())
            recent_errors = [
                e for e in self.error_history 
                if time.time() - e.get("timestamp", 0) < 3600  # Last hour
            ]
            
            return {
                "total_errors": total_errors,
                "error_counts_by_type": {k.value: v for k, v in self.error_counts.items()},
                "recent_errors_count": len(recent_errors),
                "most_common_error": max(self.error_counts.items(), key=lambda x: x[1])[0].value if self.error_counts else None,
                "error_rate": len(recent_errors) / 60 if recent_errors else 0,  # Per minute
                "system_health": self._calculate_system_health(),
                "recommendations": self._get_health_recommendations()
            }
        except Exception as e:
            logger.error(f"Failed to generate error statistics: {str(e)}")
            return {"error": "Failed to generate statistics"}
    
    def _calculate_system_health(self) -> str:
        """Calculate overall system health based on error patterns"""
        try:
            recent_errors = [
                e for e in self.error_history 
                if time.time() - e.get("timestamp", 0) < 1800  # Last 30 minutes
            ]
            
            critical_errors = sum(
                1 for e in recent_errors 
                if e.get("error_type") in [
                    ErrorType.MODEL_LOADING_ERROR.value,
                    ErrorType.CONFIGURATION_ERROR.value
                ]
            )
            
            if critical_errors > 0:
                return "critical"
            elif len(recent_errors) > 20:
                return "degraded"
            elif len(recent_errors) > 5:
                return "warning"
            else:
                return "healthy"
                
        except Exception:
            return "unknown"
    
    def _get_health_recommendations(self) -> List[str]:
        """Generate health recommendations based on error patterns"""
        try:
            recommendations = []
            
            # Check for common error patterns
            if self.error_counts.get(ErrorType.MEMORY_ERROR, 0) > 5:
                recommendations.append("Consider increasing system memory or reducing image processing batch size")
            
            if self.error_counts.get(ErrorType.TIMEOUT_ERROR, 0) > 3:
                recommendations.append("System may be overloaded - consider reducing concurrent requests")
            
            if self.error_counts.get(ErrorType.MODEL_LOADING_ERROR, 0) > 0:
                recommendations.append("Check internet connectivity and model cache directory permissions")
            
            if self.error_counts.get(ErrorType.IO_ERROR, 0) > 2:
                recommendations.append("Verify file system permissions and available disk space")
            
            recent_error_rate = len([
                e for e in self.error_history 
                if time.time() - e.get("timestamp", 0) < 300
            ])
            
            if recent_error_rate > 10:
                recommendations.append("High error rate detected - consider system restart or configuration review")
            
            if not recommendations:
                recommendations.append("System operating normally")
                
            return recommendations
            
        except Exception:
            return ["Unable to generate recommendations"]
    
    def clear_error_history(self):
        """Clear error history and reset counters"""
        self.error_history.clear()
        self.error_counts.clear()
        logger.info("Error history and counters cleared")

# Global error handler instance
error_handler = AIErrorHandler()
