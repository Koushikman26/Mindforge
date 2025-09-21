import aiohttp
import asyncio
import logging
from typing import Optional, Dict, Any, List
import os
import json
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class StockAnalysisResult:
    """Structured result from AI stock analysis"""
    success: bool
    products: Dict[str, Any]
    summary: Dict[str, Any]
    processing_time: float
    error: Optional[str] = None

class AIEngineService:
    """Service for communicating with AI engine"""
    
    def __init__(self, ai_engine_url: str = None):
        self.ai_engine_url = ai_engine_url or os.getenv("AI_ENGINE_URL", "http://ai-engine:8001")
        self._session = None
        self._last_health_check = 0
        self._health_status = None
        
    async def __aenter__(self):
        """Async context manager entry"""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=60)
        )
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._session:
            await self._session.close()
    
    async def check_health(self) -> Dict[str, Any]:
        """Check AI engine health with caching"""
        try:
            # Cache health checks for 30 seconds
            current_time = time.time()
            if (current_time - self._last_health_check) < 30 and self._health_status:
                return self._health_status
            
            if not self._session:
                self._session = aiohttp.ClientSession()
                
            async with self._session.get(f"{self.ai_engine_url}/health") as response:
                if response.status == 200:
                    result = await response.json()
                    self._health_status = result
                    self._last_health_check = current_time
                    return result
                else:
                    error_result = {
                        "status": "unhealthy", 
                        "error": f"HTTP {response.status}",
                        "timestamp": current_time
                    }
                    self._health_status = error_result
                    return error_result
                    
        except aiohttp.ClientError as e:
            logger.error(f"AI engine connection failed: {str(e)}")
            error_result = {
                "status": "unreachable", 
                "error": f"Connection failed: {str(e)}",
                "timestamp": time.time()
            }
            self._health_status = error_result
            return error_result
        except Exception as e:
            logger.error(f"AI engine health check failed: {str(e)}")
            error_result = {
                "status": "error", 
                "error": str(e),
                "timestamp": time.time()
            }
            self._health_status = error_result
            return error_result
    
    async def analyze_stock_image(
        self, 
        image_data: bytes, 
        filename: str = "image.jpg",
        single_product: Optional[str] = None
    ) -> StockAnalysisResult:
        """Send image to AI engine for stock analysis"""
        try:
            if not self._session:
                self._session = aiohttp.ClientSession()
                
            # Prepare form data
            data = aiohttp.FormData()
            data.add_field('file', image_data, filename=filename, content_type='image/jpeg')
            
            # Build URL with query parameters
            url = f"{self.ai_engine_url}/detect-stock-levels"
            if single_product:
                url += f"?single_product={single_product}"
            
            start_time = time.time()
            
            async with self._session.post(url, data=data) as response:
                processing_time = time.time() - start_time
                
                if response.status == 200:
                    result_data = await response.json()
                    
                    return StockAnalysisResult(
                        success=result_data.get("success", False),
                        products=result_data.get("stock_levels", {}),
                        summary=result_data.get("summary", {}),
                        processing_time=processing_time,
                        error=None
                    )
                    
                elif response.status == 503:
                    error_detail = await response.text()
                    return StockAnalysisResult(
                        success=False,
                        products={},
                        summary={},
                        processing_time=processing_time,
                        error="AI engine not ready"
                    )
                    
                else:
                    error_detail = await response.text()
                    logger.error(f"AI analysis failed with status {response.status}: {error_detail}")
                    
                    return StockAnalysisResult(
                        success=False,
                        products={},
                        summary={},
                        processing_time=processing_time,
                        error=f"AI engine error (HTTP {response.status}): {error_detail}"
                    )
        
        except asyncio.TimeoutError:
            logger.error("AI analysis request timed out")
            return StockAnalysisResult(
                success=False,
                products={},
                summary={},
                processing_time=60.0,  # Timeout duration
                error="Analysis request timed out"
            )
            
        except Exception as e:
            logger.error(f"AI service integration failed: {str(e)}")
            return StockAnalysisResult(
                success=False,
                products={},
                summary={},
                processing_time=0.0,
                error=f"Integration error: {str(e)}"
            )
    
    async def analyze_batch_images(
        self, 
        images: List[tuple]  # List of (filename, image_data) tuples
    ) -> Dict[str, Any]:
        """Analyze multiple images in batch"""
        try:
            if not self._session:
                self._session = aiohttp.ClientSession()
            
            if len(images) > 10:
                return {
                    "success": False,
                    "error": "Too many images (max 10 per batch)",
                    "results": []
                }
            
            # Prepare form data for batch upload
            data = aiohttp.FormData()
            for filename, image_data in images:
                data.add_field('files', image_data, filename=filename, content_type='image/jpeg')
            
            start_time = time.time()
            
            async with self._session.post(
                f"{self.ai_engine_url}/analyze-batch", 
                data=data
            ) as response:
                processing_time = time.time() - start_time
                
                if response.status == 200:
                    result_data = await response.json()
                    result_data["total_processing_time"] = processing_time
                    return result_data
                else:
                    error_detail = await response.text()
                    return {
                        "success": False,
                        "error": f"Batch analysis failed (HTTP {response.status}): {error_detail}",
                        "total_processing_time": processing_time
                    }
                    
        except Exception as e:
            logger.error(f"Batch analysis failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "total_processing_time": 0.0
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get detailed system status from AI engine"""
        try:
            if not self._session:
                self._session = aiohttp.ClientSession()
                
            async with self._session.get(f"{self.ai_engine_url}/status") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    return {
                        "error": f"Status request failed (HTTP {response.status})",
                        "available": False
                    }
                    
        except Exception as e:
            logger.error(f"System status request failed: {str(e)}")
            return {
                "error": str(e),
                "available": False
            }

# Global service instance
ai_service = AIEngineService()

# Helper function for easy usage without context manager
async def get_ai_health() -> Dict[str, Any]:
    """Quick health check without context manager"""
    async with ai_service as client:
        return await client.check_health()

async def analyze_image_quick(image_data: bytes, filename: str = "image.jpg") -> StockAnalysisResult:
    """Quick image analysis without context manager"""
    async with ai_service as client:
        return await client.analyze_stock_image(image_data, filename)
