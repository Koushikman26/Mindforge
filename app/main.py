"""
Mindforge Main Application
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Depends, Query
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
import logging
import os
import time
import tempfile
import base64
from typing import List, Optional, Dict, Any
from pathlib import Path
import aiofiles
import asyncio
import cv2
import numpy as np

# Import your existing services
from core.services.ai_service import AIEngineService, StockAnalysisResult, ai_service
from ai_engine.utils.logging_config import get_logger

# Set up logging
logger = get_logger(__name__)

# Import image processing components (initialized once at module level)
try:
    from ai_engine.image_processing.core.pipeline import AdvancedImageProcessingPipeline
    from ai_engine.image_processing.configs.processing_config import ImageProcessingConfig
    # Initialize image processing pipeline globally for reuse
    image_processing_config = ImageProcessingConfig()
    image_processing_pipeline = AdvancedImageProcessingPipeline(image_processing_config)
    IMAGE_PROCESSING_AVAILABLE = True
    logger.info("Image processing pipeline initialized successfully")
except ImportError as e:
    logger.warning(f"Image processing module not available: {e}")
    IMAGE_PROCESSING_AVAILABLE = False
    image_processing_pipeline = None

# Initialize FastAPI app
app = FastAPI(
    title="Mindforge API",
    description="AI-powered stock analysis with advanced image processing capabilities",
    version="1.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Security
security = HTTPBearer(auto_error=False)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("ALLOWED_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# Trusted host middleware
app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=os.getenv("ALLOWED_HOSTS", "*").split(",")
)

# Global variables
startup_time = time.time()
request_count = 0

# Dependency for authentication (implement based on your needs)
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Optional authentication dependency"""
    # Implement your authentication logic here
    # For now, we'll allow all requests
    return {"user_id": "anonymous"}

# Request counting middleware
@app.middleware("http")
async def count_requests(request, call_next):
    global request_count
    request_count += 1
    
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-Count"] = str(request_count)
    
    return response

# Health check endpoints
@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    try:
        # Check AI engine health
        ai_health = await ai_service.check_health()
        
        return JSONResponse({
            "status": "healthy",
            "timestamp": time.time(),
            "uptime": time.time() - startup_time,
            "request_count": request_count,
            "ai_engine": ai_health,
            "version": "1.1.0",
            "features": {
                "stock_analysis": True,
                "image_processing": True,
                "batch_processing": True,
                "multi_angle_processing": True
            }
        })
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": time.time()
            }
        )

@app.get("/status")
async def detailed_status():
    """Detailed system status including image processing pipeline"""
    try:
        async with ai_service as client:
            system_status = await client.get_system_status()
            
        return JSONResponse({
            "status": "operational",
            "timestamp": time.time(),
            "uptime": time.time() - startup_time,
            "request_count": request_count,
            "system": system_status,
            "api_version": "1.1.0"
        })
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

# Stock analysis endpoints
@app.post("/api/v1/analyze-stock")
async def analyze_stock_image(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    single_product: Optional[str] = Query(None, description="Focus analysis on a specific product"),
    enable_image_processing: bool = Query(True, description="Enable advanced image processing"),
    optimize_for_stock: bool = Query(True, description="Optimize processing for stock analysis"),
    enable_noise_reduction: bool = Query(True, description="Enable noise reduction"),
    enable_lighting_correction: bool = Query(True, description="Enable lighting correction"),
    enable_perspective_correction: bool = Query(True, description="Enable perspective correction"),
    enable_quality_enhancement: bool = Query(True, description="Enable quality enhancement"),
    current_user: dict = Depends(get_current_user)
):
    """
    Analyze stock levels in an uploaded image with optional advanced image processing
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        image_data = await file.read()
        
        if len(image_data) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        # Check file size (max 10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        if len(image_data) > max_size:
            raise HTTPException(status_code=400, detail="Image file too large (max 10MB)")
        
        # Prepare processing options
        processing_options = None
        if enable_image_processing:
            processing_options = {
                'optimize_for_stock_analysis': optimize_for_stock,
                'enable_noise_reduction': enable_noise_reduction,
                'enable_lighting_correction': enable_lighting_correction,
                'enable_perspective_correction': enable_perspective_correction,
                'enable_quality_enhancement': enable_quality_enhancement,
                'preserve_details': True,
                'save_intermediate_steps': False
            }
        
        # Analyze image
        async with ai_service as client:
            result = await client.analyze_stock_image(
                image_data=image_data,
                filename=file.filename or "uploaded_image.jpg",
                single_product=single_product,
                enable_image_processing=enable_image_processing,
                processing_options=processing_options
            )
        
        # Prepare response
        response_data = {
            "success": result.success,
            "products": result.products,
            "summary": result.summary,
            "processing_time": result.processing_time,
            "metadata": {
                "filename": file.filename,
                "file_size": len(image_data),
                "content_type": file.content_type,
                "single_product_mode": single_product is not None,
                "image_processing_enabled": enable_image_processing
            }
        }
        
        # Add image processing metadata if available
        if result.image_processing_time is not None:
            response_data["image_processing"] = {
                "processing_time": result.image_processing_time,
                "improvements": result.image_improvements,
                "processing_steps": result.processing_steps
            }
        
        if result.error:
            response_data["error"] = result.error
        
        return JSONResponse(response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Stock analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/api/v1/analyze-batch")
async def analyze_batch_images(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    enable_image_processing: bool = Query(True, description="Enable advanced image processing"),
    optimize_for_stock: bool = Query(True, description="Optimize processing for stock analysis"),
    current_user: dict = Depends(get_current_user)
):
    """
    Analyze multiple images in batch with optional multi-angle processing
    """
    try:
        # Validate batch size
        if len(files) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 images allowed per batch")
        
        if len(files) == 0:
            raise HTTPException(status_code=400, detail="At least one image is required")
        
        # Prepare images
        images = []
        total_size = 0
        
        for file in files:
            # Validate file type
            if not file.content_type or not file.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=400, 
                    detail=f"File {file.filename} is not an image"
                )
            
            # Read image data
            image_data = await file.read()
            
            if len(image_data) == 0:
                raise HTTPException(
                    status_code=400, 
                    detail=f"File {file.filename} is empty"
                )
            
            total_size += len(image_data)
            images.append((file.filename or f"image_{len(images)}.jpg", image_data))
        
        # Check total batch size (max 50MB)
        max_batch_size = 50 * 1024 * 1024  # 50MB
        if total_size > max_batch_size:
            raise HTTPException(status_code=400, detail="Batch size too large (max 50MB total)")
        
        # Prepare processing options
        processing_options = None
        if enable_image_processing:
            processing_options = {
                'optimize_for_stock_analysis': optimize_for_stock,
                'enable_noise_reduction': True,
                'enable_lighting_correction': True,
                'enable_perspective_correction': True,
                'enable_quality_enhancement': True,
                'preserve_details': True,
                'save_intermediate_steps': False
            }
        
        # Analyze batch
        async with ai_service as client:
            result = await client.analyze_batch_images(
                images=images,
                enable_image_processing=enable_image_processing,
                processing_options=processing_options
            )
        
        # Add metadata
        result["metadata"] = {
            "batch_size": len(images),
            "total_file_size": total_size,
            "image_processing_enabled": enable_image_processing,
            "filenames": [filename for filename, _ in images]
        }
        
        return JSONResponse(result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

# Image processing endpoints
@app.post("/api/v1/process-image")
async def process_single_image(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    enable_noise_reduction: bool = Query(True, description="Enable noise reduction"),
    enable_lighting_correction: bool = Query(True, description="Enable lighting correction"),
    enable_perspective_correction: bool = Query(True, description="Enable perspective correction"),
    enable_quality_enhancement: bool = Query(True, description="Enable quality enhancement"),
    optimize_for_stock_analysis: bool = Query(True, description="Optimize for stock analysis"),
    preserve_details: bool = Query(True, description="Preserve fine details"),
    return_base64: bool = Query(True, description="Return processed image as base64"),
    current_user: dict = Depends(get_current_user)
):
    """
    Process a single image through the advanced image processing pipeline
    """
    try:
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read image data
        image_data = await file.read()
        
        if len(image_data) == 0:
            raise HTTPException(status_code=400, detail="Empty image file")
        
        # Check file size (max 10MB)
        max_size = 10 * 1024 * 1024  # 10MB
        if len(image_data) > max_size:
            raise HTTPException(status_code=400, detail="Image file too large (max 10MB)")
        
        # Prepare processing options
        processing_options = {
            'enable_noise_reduction': enable_noise_reduction,
            'enable_lighting_correction': enable_lighting_correction,
            'enable_perspective_correction': enable_perspective_correction,
            'enable_quality_enhancement': enable_quality_enhancement,
            'optimize_for_stock_analysis': optimize_for_stock_analysis,
            'preserve_details': preserve_details,
            'save_intermediate_steps': False
        }
        
        # Process image
        async with ai_service as client:
            result = await client.process_single_image_only(
                image_data=image_data,
                processing_options=processing_options
            )
        
        # Prepare response
        response_data = {
            "success": result['success'],
            "processing_time": result['processing_time'],
            "metadata": {
                "filename": file.filename,
                "file_size": len(image_data),
                "content_type": file.content_type,
                "processing_options": processing_options
            }
        }
        
        if result['success']:
            response_data.update({
                "processing_steps": result.get('processing_steps', []),
                "improvements": result.get('improvements', {}),
                "warnings": result.get('warnings', [])
            })
            
            # Include processed image if requested
            if return_base64 and 'processed_image_bytes' in result:
                img_base64 = base64.b64encode(result['processed_image_bytes']).decode('utf-8')
                response_data["processed_image"] = img_base64
        else:
            response_data["error"] = result.get('error', 'Unknown processing error')
        
        return JSONResponse(response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

@app.post("/api/v1/process-multiple-images")
async def process_multiple_images(
    files: List[UploadFile] = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks(),
    optimize_for_stock_analysis: bool = Query(True, description="Optimize for stock analysis"),
    return_composite: bool = Query(True, description="Return composite image from best results"),
    current_user: dict = Depends(get_current_user)
):
    """
    Process multiple images with multi-angle support
    """
    try:
        # Validate batch size
        if len(files) > 10:
            raise HTTPException(status_code=400, detail="Maximum 10 images allowed")
        
        if len(files) == 0:
            raise HTTPException(status_code=400, detail="At least one image is required")
        
        # Prepare images
        images = []
        total_size = 0
        
        for file in files:
            # Validate file type
            if not file.content_type or not file.content_type.startswith('image/'):
                raise HTTPException(
                    status_code=400, 
                    detail=f"File {file.filename} is not an image"
                )
            
            # Read image data
            image_data = await file.read()
            
            if len(image_data) == 0:
                raise HTTPException(
                    status_code=400, 
                    detail=f"File {file.filename} is empty"
                )
            
            total_size += len(image_data)
            images.append(image_data)
        
        # Check total batch size (max 50MB)
        max_batch_size = 50 * 1024 * 1024  # 50MB
        if total_size > max_batch_size:
            raise HTTPException(status_code=400, detail="Batch size too large (max 50MB total)")
        
        # Prepare processing options
        processing_options = {
            'optimize_for_stock_analysis': optimize_for_stock_analysis,
            'save_intermediate_steps': False
        }
        
        # Check if image processing is available
        if not IMAGE_PROCESSING_AVAILABLE:
            raise HTTPException(status_code=503, detail="Image processing module is not available")

        # Use the global pipeline instance

        # Convert bytes to numpy arrays
        image_arrays = []
        for image_data in images:
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if image is not None:
                image_arrays.append(image)
        
        if not image_arrays:
            raise HTTPException(status_code=400, detail="No valid images could be decoded")
        
        # Process in executor to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            image_processing_pipeline.process_multiple_images,
            image_arrays,
            processing_options
        )
        
        # Prepare response
        response_data = {
            "success": result['successful_count'] > 0,
            "successful_count": result['successful_count'],
            "failed_count": result['failed_count'],
            "total_processing_time": result['total_processing_time'],
            "average_improvements": result.get('average_improvements', {}),
            "metadata": {
                "batch_size": len(files),
                "total_file_size": total_size,
                "filenames": [f.filename for f in files]
            }
        }
        
        # Include composite image if available and requested
        if return_composite and result.get('composite_result'):
            composite = result['composite_result']
            if 'selected_image' in composite:
                # Convert composite image to base64
                _, buffer = cv2.imencode('.jpg', composite['selected_image'])
                composite_base64 = base64.b64encode(buffer).decode('utf-8')
                response_data["composite_image"] = composite_base64
                response_data["composite_metadata"] = {
                    "method": composite.get('method', 'unknown'),
                    "quality_score": composite.get('quality_score', 0),
                    "total_candidates": composite.get('total_candidates', 0)
                }
        
        return JSONResponse(response_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multi-image processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Multi-image processing failed: {str(e)}")

# Statistics and monitoring endpoints
@app.get("/api/v1/processing-statistics")
async def get_processing_statistics(current_user: dict = Depends(get_current_user)):
    """
    Get comprehensive processing statistics
    """
    try:
        async with ai_service as client:
            stats = await client.get_image_processing_statistics()
        
        # Add API-level statistics
        stats["api_statistics"] = {
            "total_requests": request_count,
            "uptime_seconds": time.time() - startup_time,
            "requests_per_minute": request_count / max((time.time() - startup_time) / 60, 1)
        }
        
        return JSONResponse(stats)
        
    except Exception as e:
        logger.error(f"Failed to get processing statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Statistics unavailable: {str(e)}")

@app.post("/api/v1/benchmark-pipeline")
async def benchmark_pipeline(
    iterations: int = Query(3, ge=1, le=10, description="Number of benchmark iterations"),
    image_count: int = Query(5, ge=1, le=20, description="Number of test images"),
    current_user: dict = Depends(get_current_user)
):
    """
    Run performance benchmark on the image processing pipeline
    """
    try:
        if iterations > 10 or image_count > 20:
            raise HTTPException(
                status_code=400, 
                detail="Benchmark parameters too large (max 10 iterations, 20 images)"
            )
        
        async with ai_service as client:
            benchmark_result = await client.benchmark_image_processing(
                test_image_count=image_count,
                iterations=iterations
            )
        
        if benchmark_result['success']:
            return JSONResponse({
                "benchmark_completed": True,
                "parameters": {
                    "iterations": iterations,
                    "image_count": image_count
                },
                "results": benchmark_result['benchmark_results']
            })
        else:
            raise HTTPException(
                status_code=500, 
                detail=f"Benchmark failed: {benchmark_result.get('error', 'Unknown error')}"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Benchmark failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")

# Legacy compatibility endpoints
@app.post("/detect-stock-levels")
async def detect_stock_levels_legacy(
    file: UploadFile = File(...),
    single_product: Optional[str] = Query(None),
    current_user: dict = Depends(get_current_user)
):
    """
    Legacy endpoint for backward compatibility
    """
    return await analyze_stock_image(
        file=file,
        background_tasks=BackgroundTasks(),
        single_product=single_product,
        enable_image_processing=True,
        current_user=current_user
    )

@app.post("/analyze-batch")
async def analyze_batch_legacy(
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user)
):
    """
    Legacy batch analysis endpoint for backward compatibility
    """
    return await analyze_batch_images(
        files=files,
        background_tasks=BackgroundTasks(),
        enable_image_processing=True,
        current_user=current_user
    )

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": time.time()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": time.time()
        }
    )

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    global startup_time
    startup_time = time.time()
    
    logger.info("Starting Mindforge API with advanced image processing...")
    logger.info(f"API version: 1.1.0")
    logger.info(f"AI Engine URL: {os.getenv('AI_ENGINE_URL', 'http://ai-engine:8001')}")
    
    # Warm up the AI service
    try:
        health = await ai_service.check_health()
        logger.info(f"AI Engine status: {health.get('status', 'unknown')}")
    except Exception as e:
        logger.warning(f"AI Engine not immediately available: {str(e)}")

@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks"""
    logger.info("Shutting down Mindforge API...")
    
    # Close AI service connections
    try:
        if ai_service._session:
            await ai_service._session.close()
    except Exception as e:
        logger.error(f"Error closing AI service connections: {str(e)}")

# Root endpoint
@app.get("/")
async def root():
    """API root endpoint with feature overview"""
    return JSONResponse({
        "name": "Mindforge API",
        "version": "1.1.0",
        "description": "AI-powered stock analysis with advanced image processing",
        "features": {
            "stock_analysis": {
                "description": "Analyze stock levels from images",
                "endpoints": ["/api/v1/analyze-stock", "/api/v1/analyze-batch"]
            },
            "image_processing": {
                "description": "Advanced image processing pipeline",
                "endpoints": ["/api/v1/process-image", "/api/v1/process-multiple-images"],
                "capabilities": [
                    "automatic_lighting_correction",
                    "perspective_transformation",
                    "quality_enhancement",
                    "noise_reduction",
                    "multi_angle_processing"
                ]
            },
            "monitoring": {
                "description": "System monitoring and statistics",
                "endpoints": ["/health", "/status", "/api/v1/processing-statistics"]
            }
        },
        "documentation": {
            "swagger_ui": "/docs",
            "redoc": "/redoc"
        },
        "uptime": time.time() - startup_time,
        "requests_served": request_count
    })

# Development server
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=os.getenv("ENVIRONMENT", "production") == "development",
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )