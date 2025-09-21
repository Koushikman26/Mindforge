import aiohttp
import asyncio
import logging
from typing import Optional, Dict, Any, List, Union
import os
import json
import time
import tempfile
import cv2
import numpy as np
from pathlib import Path
from dataclasses import dataclass

# Import the image processing pipeline
from ai_engine.image_processing.core.pipeline import AdvancedImageProcessingPipeline
from ai_engine.image_processing.configs.processing_config import ImageProcessingConfig

logger = logging.getLogger(__name__)

@dataclass
class StockAnalysisResult:
    """Structured result from AI stock analysis"""
    success: bool
    products: Dict[str, Any]
    summary: Dict[str, Any]
    processing_time: float
    error: Optional[str] = None
    # New fields for image processing integration
    image_processing_time: Optional[float] = None
    image_improvements: Optional[Dict[str, Any]] = None
    processing_steps: Optional[List[str]] = None

class AIEngineService:
    """Service for communicating with AI engine with integrated image processing"""
    
    def __init__(self, ai_engine_url: str = None):
        self.ai_engine_url = ai_engine_url or os.getenv("AI_ENGINE_URL", "http://ai-engine:8001")
        self._session = None
        self._last_health_check = 0
        self._health_status = None
        
        # Initialize image processing pipeline
        self.image_config = ImageProcessingConfig()
        self.image_pipeline = AdvancedImageProcessingPipeline(self.image_config)
        
        logger.info("AIEngineService initialized with advanced image processing pipeline")
        
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
                    # Add image processing pipeline status
                    result["image_processing"] = {
                        "enabled": True,
                        "pipeline_version": "1.0.0",
                        "processing_stats": self.image_pipeline.get_processing_statistics()
                    }
                    self._health_status = result
                    self._last_health_check = current_time
                    return result
                else:
                    error_result = {
                        "status": "unhealthy", 
                        "error": f"HTTP {response.status}",
                        "timestamp": current_time,
                        "image_processing": {"enabled": True, "status": "ready"}
                    }
                    self._health_status = error_result
                    return error_result
                    
        except aiohttp.ClientError as e:
            logger.error(f"AI engine connection failed: {str(e)}")
            error_result = {
                "status": "unreachable", 
                "error": f"Connection failed: {str(e)}",
                "timestamp": time.time(),
                "image_processing": {"enabled": True, "status": "ready"}
            }
            self._health_status = error_result
            return error_result
        except Exception as e:
            logger.error(f"AI engine health check failed: {str(e)}")
            error_result = {
                "status": "error", 
                "error": str(e),
                "timestamp": time.time(),
                "image_processing": {"enabled": True, "status": "ready"}
            }
            self._health_status = error_result
            return error_result

    async def _process_image_async(
        self, 
        image_data: Union[bytes, np.ndarray], 
        processing_options: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Async wrapper for image processing pipeline
        
        Args:
            image_data: Raw image bytes or numpy array
            processing_options: Processing configuration options
            
        Returns:
            Dictionary with processing results
        """
        try:
            # Default processing options optimized for stock analysis
            default_options = {
                'optimize_for_stock_analysis': True,
                'enable_noise_reduction': True,
                'enable_lighting_correction': True,
                'enable_perspective_correction': True,
                'enable_quality_enhancement': True,
                'preserve_details': True,
                'save_intermediate_steps': False  # Don't save intermediate steps in production
            }
            
            if processing_options:
                default_options.update(processing_options)
            
            # Run image processing in executor to avoid blocking
            loop = asyncio.get_event_loop()
            
            if isinstance(image_data, bytes):
                # Convert bytes to numpy array
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if image is None:
                    return {
                        'success': False,
                        'error': 'Failed to decode image data',
                        'processing_time': 0.0
                    }
                # Process image using the pipeline
                return await loop.run_in_executor(
                    None, 
                    self.image_pipeline.process_single_image,
                    image,
                    default_options
                )
            # Image is already a numpy array
            return await loop.run_in_executor(
                None,
                self.image_pipeline.process_single_image,
                image_data,
                default_options
            )
            
        except Exception as e:
            logger.error(f"Image processing failed: {str(e)}")
            return {
                'success': False,
                'error': f'Image processing error: {str(e)}',
                'processing_time': 0.0
            }

    async def _convert_processed_image_to_bytes(self, processed_image: np.ndarray) -> bytes:
        """Convert processed numpy array back to bytes for transmission"""
        try:
            # Encode image as JPEG bytes
            loop = asyncio.get_event_loop()
            
            def encode_image():
                _, buffer = cv2.imencode('.jpg', processed_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
                return buffer.tobytes()
            
            return await loop.run_in_executor(None, encode_image)
            
        except Exception as e:
            logger.error(f"Failed to convert processed image to bytes: {str(e)}")
            raise
    
    async def analyze_stock_image(
        self, 
        image_data: bytes, 
        filename: str = "image.jpg",
        single_product: Optional[str] = None,
        enable_image_processing: bool = True,
        processing_options: Optional[Dict] = None
    ) -> StockAnalysisResult:
        """
        Send image to AI engine for stock analysis with optional image processing
        
        Args:
            image_data: Raw image bytes
            filename: Original filename
            single_product: Optional single product detection
            enable_image_processing: Whether to apply image processing pipeline
            processing_options: Image processing configuration
        """
        try:
            if not self._session:
                self._session = aiohttp.ClientSession()
            
            start_time = time.time()
            processed_image_data = image_data
            image_processing_result = None
            
            # Apply image processing if enabled
            if enable_image_processing:
                logger.info("Applying image processing pipeline for stock analysis")
                
                image_processing_result = await self._process_image_async(
                    image_data, 
                    processing_options
                )
                
                if image_processing_result['success']:
                    # Convert processed image back to bytes
                    processed_image_data = await self._convert_processed_image_to_bytes(
                        image_processing_result['processed_image']
                    )
                    logger.info(f"Image processing completed in {image_processing_result['processing_time']:.3f}s")
                else:
                    logger.warning(f"Image processing failed: {image_processing_result['error']}")
                    # Continue with original image
                    processed_image_data = image_data
            
            # Prepare form data with processed image
            data = aiohttp.FormData()
            data.add_field('file', processed_image_data, filename=filename, content_type='image/jpeg')
            
            # Build URL with query parameters
            url = f"{self.ai_engine_url}/detect-stock-levels"
            if single_product:
                url += f"?single_product={single_product}"
            
            # Send to AI engine for analysis
            async with self._session.post(url, data=data) as response:
                total_processing_time = time.time() - start_time
                
                if response.status == 200:
                    result_data = await response.json()
                    
                    return StockAnalysisResult(
                        success=result_data.get("success", False),
                        products=result_data.get("stock_levels", {}),
                        summary=result_data.get("summary", {}),
                        processing_time=total_processing_time,
                        error=None,
                        # Add image processing metadata
                        image_processing_time=(
                            image_processing_result['processing_time'] 
                            if image_processing_result and image_processing_result['success'] 
                            else None
                        ),
                        image_improvements=(
                            image_processing_result['improvements'] 
                            if image_processing_result and image_processing_result['success'] 
                            else None
                        ),
                        processing_steps=(
                            image_processing_result['processing_steps'] 
                            if image_processing_result and image_processing_result['success'] 
                            else None
                        )
                    )
                    
                elif response.status == 503:
                    error_detail = await response.text()
                    return StockAnalysisResult(
                        success=False,
                        products={},
                        summary={},
                        processing_time=total_processing_time,
                        error="AI engine not ready",
                        image_processing_time=(
                            image_processing_result['processing_time'] 
                            if image_processing_result 
                            else None
                        )
                    )
                    
                else:
                    error_detail = await response.text()
                    logger.error(f"AI analysis failed with status {response.status}: {error_detail}")
                    
                    return StockAnalysisResult(
                        success=False,
                        products={},
                        summary={},
                        processing_time=total_processing_time,
                        error=f"AI engine error (HTTP {response.status}): {error_detail}",
                        image_processing_time=(
                            image_processing_result['processing_time'] 
                            if image_processing_result 
                            else None
                        )
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
        images: List[tuple],  # List of (filename, image_data) tuples
        enable_image_processing: bool = True,
        processing_options: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Analyze multiple images in batch with optional image processing
        
        Args:
            images: List of (filename, image_data) tuples
            enable_image_processing: Whether to apply image processing pipeline
            processing_options: Image processing configuration
        """
        try:
            if not self._session:
                self._session = aiohttp.ClientSession()
            
            if len(images) > 10:
                return {
                    "success": False,
                    "error": "Too many images (max 10 per batch)",
                    "results": []
                }
            
            start_time = time.time()
            processed_images = []
            image_processing_results = []
            
            # Process images if enabled
            if enable_image_processing:
                logger.info(f"Processing {len(images)} images through processing pipeline")
                
                # Process all images through the pipeline
                image_data_list = [img_data for _, img_data in images]
                
                # Use multi-image processing for better results
                loop = asyncio.get_event_loop()
                multi_processing_result = await loop.run_in_executor(
                    None,
                    self.image_pipeline.process_multiple_images,
                    image_data_list,
                    processing_options
                )
                
                # Convert processed images back to bytes
                for idx, (filename, _) in enumerate(images):
                    if idx < len(multi_processing_result['individual_results']):
                        individual_result = multi_processing_result['individual_results'][idx]
                        
                        if individual_result['success']:
                            processed_bytes = await self._convert_processed_image_to_bytes(
                                individual_result['processed_image']
                            )
                            processed_images.append((filename, processed_bytes))
                        else:
                            # Use original image if processing failed
                            processed_images.append((filename, images[idx][1]))
                        image_processing_results.append(individual_result)
                    else:
                        # Use original image if not processed
                        processed_images.append((filename, images[idx][1]))
                        image_processing_results.append({"success": False, "error": "Not processed"})
            else:
                processed_images = images
                image_processing_results = []
            
            # Prepare form data for batch upload
            data = aiohttp.FormData()
            for filename, image_data in processed_images:
                data.add_field('files', image_data, filename=filename, content_type='image/jpeg')
            
            # Send to AI engine
            async with self._session.post(
                f"{self.ai_engine_url}/analyze-batch", 
                data=data
            ) as response:
                total_processing_time = time.time() - start_time
                
                if response.status == 200:
                    result_data = await response.json()
                    result_data["total_processing_time"] = total_processing_time
                    
                    # Add image processing metadata
                    if enable_image_processing and image_processing_results:
                        result_data["image_processing"] = {
                            "enabled": True,
                            "results": image_processing_results,
                            "multi_angle_result": multi_processing_result if 'multi_processing_result' in locals() else None
                        }
                    else:
                        result_data["image_processing"] = {"enabled": False}
                    
                    return result_data
                else:
                    error_detail = await response.text()
                    return {
                        "success": False,
                        "error": f"Batch analysis failed (HTTP {response.status}): {error_detail}",
                        "total_processing_time": total_processing_time,
                        "image_processing": {
                            "enabled": enable_image_processing,
                            "results": image_processing_results if enable_image_processing else []
                        }
                    }
                    
        except Exception as e:
            logger.error(f"Batch analysis failed: {str(e)}")
            return {
                "success": False,
                "error": str(e),
                "total_processing_time": 0.0,
                "image_processing": {"enabled": enable_image_processing, "error": str(e)}
            }

    async def process_single_image_only(
        self,
        image_data: bytes,
        processing_options: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Process a single image through the pipeline without AI analysis
        
        Args:
            image_data: Raw image bytes
            processing_options: Processing configuration options
            
        Returns:
            Image processing results
        """
        try:
            result = await self._process_image_async(image_data, processing_options)
            
            if result['success']:
                # Convert processed image to bytes for response
                processed_bytes = await self._convert_processed_image_to_bytes(
                    result['processed_image']
                )
                result['processed_image_bytes'] = processed_bytes
                # Remove the numpy array from response to make it JSON serializable
                del result['processed_image']
            
            return result
            
        except Exception as e:
            logger.error(f"Image processing only failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': 0.0
            }

    async def get_image_processing_statistics(self) -> Dict[str, Any]:
        """Get image processing pipeline statistics"""
        try:
            stats = self.image_pipeline.get_processing_statistics()
            return {
                "success": True,
                "statistics": stats,
                "pipeline_config": {
                    "max_processing_time": self.image_config.MAX_PROCESSING_TIME,
                    "target_size": self.image_config.TARGET_SIZE,
                    "enable_shelf_detection": self.image_config.ENABLE_SHELF_DETECTION,
                    "enable_product_segmentation": self.image_config.ENABLE_PRODUCT_SEGMENTATION
                }
            }
        except Exception as e:
            logger.error(f"Failed to get image processing statistics: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }

    async def benchmark_image_processing(
        self,
        test_image_count: int = 5,
        iterations: int = 3
    ) -> Dict[str, Any]:
        """
        Benchmark the image processing pipeline
        
        Args:
            test_image_count: Number of test images to generate
            iterations: Number of iterations to run
            
        Returns:
            Benchmark results
        """
        try:
            # Generate test images
            test_images = []
            for _ in range(test_image_count):
                # Create a random test image
                test_image = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
                test_images.append(test_image)
            
            # Run benchmark in executor
            loop = asyncio.get_event_loop()
            benchmark_result = await loop.run_in_executor(
                None,
                self.image_pipeline.benchmark_pipeline,
                test_images,
                iterations
            )
            
            return {
                "success": True,
                "benchmark_results": benchmark_result
            }
            
        except Exception as e:
            logger.error(f"Image processing benchmark failed: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get detailed system status from AI engine including image processing"""
        try:
            if not self._session:
                self._session = aiohttp.ClientSession()
                
            async with self._session.get(f"{self.ai_engine_url}/status") as response:
                if response.status == 200:
                    ai_status = await response.json()
                else:
                    ai_status = {
                        "error": f"Status request failed (HTTP {response.status})",
                        "available": False
                    }
                
                # Add image processing status
                image_processing_stats = await self.get_image_processing_statistics()
                
                return {
                    "ai_engine": ai_status,
                    "image_processing": image_processing_stats,
                    "integration": {
                        "version": "1.0.0",
                        "features": [
                            "automatic_lighting_correction",
                            "perspective_transformation", 
                            "quality_enhancement",
                            "noise_reduction",
                            "multi_angle_processing"
                        ]
                    }
                }
                    
        except Exception as e:
            logger.error(f"System status request failed: {str(e)}")
            return {
                "ai_engine": {"error": str(e), "available": False},
                "image_processing": {"error": str(e), "available": False}
            }

# Global service instance
ai_service = AIEngineService()

# Helper functions for easy usage without context manager
async def get_ai_health() -> Dict[str, Any]:
    """Quick health check without context manager"""
    async with ai_service as client:
        return await client.check_health()

async def analyze_image_quick(
    image_data: bytes, 
    filename: str = "image.jpg",
    enable_image_processing: bool = True
) -> StockAnalysisResult:
    """Quick image analysis without context manager"""
    async with ai_service as client:
        return await client.analyze_stock_image(
            image_data, 
            filename, 
            enable_image_processing=enable_image_processing
        )

async def process_image_only(
    image_data: bytes,
    processing_options: Optional[Dict] = None
) -> Dict[str, Any]:
    """Quick image processing without context manager"""
    async with ai_service as client:
        return await client.process_single_image_only(image_data, processing_options)