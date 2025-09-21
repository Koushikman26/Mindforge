"""
Image Processing Pipeline for Stock Analysis

This module provides advanced image processing capabilities for stock level analysis,
including noise reduction, lighting correction, perspective transformation, and quality enhancement.
"""

from .core.pipeline import AdvancedImageProcessingPipeline
from .configs.processing_config import ImageProcessingConfig

__all__ = [
    'AdvancedImageProcessingPipeline',
    'ImageProcessingConfig',
]

__version__ = '1.0.0'
__author__ = 'Mindforge Team'