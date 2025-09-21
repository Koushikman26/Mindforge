import cv2
import numpy as np
import time
import os
from typing import Dict, List, Optional, Union, Any
from pathlib import Path

from .lighting_correction import AdvancedLightingCorrector
from .perspective_transform import PerspectiveTransformer
from .quality_enhancement import QualityEnhancer
from .noise_reduction import AdvancedNoiseReducer
from ..utils.image_utils import (
    timing_decorator, validate_image, load_image, save_image, 
    calculate_image_metrics, logger, ImageProcessingError, ProcessingTimeoutError
)
from ..configs.processing_config import ImageProcessingConfig

class AdvancedImageProcessingPipeline:
    """Complete advanced image processing pipeline for Mindforge stock analysis"""
    
    def __init__(self, config: Optional[ImageProcessingConfig] = None):
        self.config = config or ImageProcessingConfig()
        
        # Initialize all processing modules
        self.lighting_corrector = AdvancedLightingCorrector(self.config)
        self.perspective_transformer = PerspectiveTransformer(self.config)
        self.quality_enhancer = QualityEnhancer(self.config)
        self.noise_reducer = AdvancedNoiseReducer(self.config)
        
        # Processing statistics
        self.stats = {
            'total_processed': 0,
            'successful_processed': 0,
            'average_processing_time': 0.0,
            'total_processing_time': 0.0,
            'error_count': 0,
            'last_error': None
        }
        
        logger.info(f"AdvancedImageProcessingPipeline initialized with config: {self.config}")
    
    @timing_decorator
    def process_single_image(self, 
                           image_input: Union[np.ndarray, str, Path], 
                           options: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process a single image through the complete pipeline
        
        Args:
            image_input: Input image as numpy array or file path
            options: Processing options and parameters
            
        Returns:
            Complete processing results with metadata
        """
        start_time = time.time()
        self.stats['total_processed'] += 1
        
        # Parse options
        opts = self._parse_processing_options(options)
        source_path = "unknown"  # Initialize to avoid UnboundLocalError

        try:
            # Load image if path provided
            if isinstance(image_input, (str, Path)):
                image = load_image(image_input)
                if image is None:
                    raise ImageProcessingError(f"Failed to load image from {image_input}")
                source_path = str(image_input)
            else:
                image = image_input.copy() if image_input is not None else None
                source_path = "numpy_array"
            
            if not validate_image(image):
                raise ImageProcessingError("Invalid input image")
            
            # Initialize result structure
            result = self._initialize_result_structure(image, source_path, opts)
            
            # Apply processing pipeline
            processed_image = self._execute_pipeline(image, opts, result)
            
            # Finalize results
            processing_time = time.time() - start_time
            self._finalize_results(result, processed_image, processing_time, opts)
            
            # Update statistics
            self.stats['successful_processed'] += 1
            self._update_processing_stats(processing_time, success=True)
            
            logger.info(f"Successfully processed image in {processing_time:.3f}s")
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_result = self._handle_processing_error(e, processing_time, source_path)
            self._update_processing_stats(processing_time, success=False, error=str(e))
            return error_result
    
    def _parse_processing_options(self, options: Optional[Dict]) -> Dict:
        """Parse and validate processing options"""
        default_options = {
            # Pipeline control
            'enable_noise_reduction': True,
            'enable_lighting_correction': True,
            'enable_perspective_correction': True,
            'enable_rotation_correction': True,
            'enable_quality_enhancement': True,
            
            # Processing parameters
            'noise_reduction_level': 'auto',
            'lighting_correction_type': 'auto',
            'enhancement_level': 'auto',
            'preserve_details': True,
            
            # Output control
            'save_intermediate_steps': self.config.SAVE_INTERMEDIATE_RESULTS,
            'output_directory': self.config.OUTPUT_DIR,
            'output_format': 'jpg',
            'output_quality': 95,
            
            # Performance
            'timeout_seconds': self.config.MAX_PROCESSING_TIME,
            'enable_validation': True,
            
            # Stock analysis specific
            'optimize_for_stock_analysis': True,
            'enhance_text_readability': True,
            'enhance_product_visibility': True
        }
        
        if options:
            default_options.update(options)
        
        return default_options
    
    def _initialize_result_structure(self, image: np.ndarray, source_path: str, opts: Dict) -> Dict:
        """Initialize the result data structure"""
        original_metrics = calculate_image_metrics(image)
        
        return {
            'source_path': source_path,
            'original_image': image.copy(),
            'processed_image': None,
            'intermediate_images': {},
            'processing_steps': [],
            'original_metrics': original_metrics,
            'final_metrics': None,
            'improvements': {},
            'processing_options': opts.copy(),
            'processing_time': 0.0,
            'success': False,
            'error': None,
            'warnings': [],
            'pipeline_version': '1.0.0'
        }
    
    def _execute_pipeline(self, image: np.ndarray, opts: Dict, result: Dict) -> np.ndarray:
        """Execute the main processing pipeline"""
        current_image = image.copy()
        step_number = 1
        
        # Step 1: Noise Reduction
        if opts['enable_noise_reduction']:
            logger.info(f"Step {step_number}: Noise Reduction")
            current_image = self.noise_reducer.reduce_noise(
                current_image, 
                noise_type=opts['noise_reduction_level'],
                preserve_details=opts['preserve_details']
            )
            self._save_intermediate_step(current_image, 'noise_reduction', opts, result)
            result['processing_steps'].append('noise_reduction')
            step_number += 1
        
        # Step 2: Lighting Correction
        if opts['enable_lighting_correction']:
            logger.info(f"Step {step_number}: Lighting Correction")
            if opts['optimize_for_stock_analysis']:
                current_image = self.lighting_corrector.correct_shelf_lighting(current_image)
            else:
                current_image = self.lighting_corrector.correct_lighting(
                    current_image, 
                    correction_type=opts['lighting_correction_type']
                )
            self._save_intermediate_step(current_image, 'lighting_correction', opts, result)
            result['processing_steps'].append('lighting_correction')
            step_number += 1
        
        # Step 3: Perspective Correction
        if opts['enable_perspective_correction']:
            logger.info(f"Step {step_number}: Perspective Correction")
            current_image = self.perspective_transformer.correct_ceiling_perspective(
                current_image,
                shelf_detection=opts['optimize_for_stock_analysis']
            )
            self._save_intermediate_step(current_image, 'perspective_correction', opts, result)
            result['processing_steps'].append('perspective_correction')
            step_number += 1
        
        # Step 4: Rotation Correction
        if opts['enable_rotation_correction']:
            logger.info(f"Step {step_number}: Rotation Correction")
            current_image = self.perspective_transformer.correct_rotation(current_image)
            
            # Auto-crop if needed
            current_image = self.perspective_transformer.auto_crop_rotated_image(current_image)
            
            self._save_intermediate_step(current_image, 'rotation_correction', opts, result)
            result['processing_steps'].append('rotation_correction')
            step_number += 1
        
        # Step 5: Quality Enhancement
        if opts['enable_quality_enhancement']:
            logger.info(f"Step {step_number}: Quality Enhancement")
            if opts['optimize_for_stock_analysis']:
                current_image = self.quality_enhancer.enhance_for_stock_analysis(current_image)
            else:
                current_image = self.quality_enhancer.enhance_image_quality(
                    current_image,
                    enhancement_level=opts['enhancement_level']
                )
            self._save_intermediate_step(current_image, 'quality_enhancement', opts, result)
            result['processing_steps'].append('quality_enhancement')
            step_number += 1
        
        return current_image
    
    def _save_intermediate_step(self, image: np.ndarray, step_name: str, opts: Dict, result: Dict):
        """Save intermediate processing step if enabled"""
        if opts['save_intermediate_steps']:
            result['intermediate_images'][step_name] = image.copy()
            
            # Also save to disk if output directory is specified
            if opts['output_directory']:
                timestamp = int(time.time())
                filename = f"intermediate_{step_name}_{timestamp}.{opts['output_format']}"
                filepath = Path(opts['output_directory']) / filename
                
                if save_image(image, filepath, opts['output_quality']):
                    logger.debug(f"Saved intermediate step {step_name} to {filepath}")
    
    def _finalize_results(self, result: Dict, processed_image: np.ndarray, 
                         processing_time: float, opts: Dict):
        """Finalize processing results with metrics and analysis"""
        result['processed_image'] = processed_image
        result['processing_time'] = processing_time
        result['success'] = True
        
        # Calculate final metrics
        result['final_metrics'] = calculate_image_metrics(processed_image)
        
        # Calculate improvements
        result['improvements'] = self._calculate_improvements(
            result['original_metrics'], 
            result['final_metrics']
        )
        
        # Performance validation
        if processing_time > self.config.MAX_PROCESSING_TIME:
            warning = f"Processing time ({processing_time:.2f}s) exceeded target ({self.config.MAX_PROCESSING_TIME}s)"
            result['warnings'].append(warning)
            logger.warning(warning)
        
        # Quality validation
        if opts['enable_validation']:
            quality_warnings = self._validate_processing_quality(result)
            result['warnings'].extend(quality_warnings)
        
        logger.info(f"Processing completed: {len(result['processing_steps'])} steps, "
                   f"{len(result['warnings'])} warnings")
    
    def _calculate_improvements(self, original_metrics: Dict, final_metrics: Dict) -> Dict:
        """Calculate quantitative improvements from processing"""
        improvements = {}
        
        for metric in ['mean_brightness', 'contrast', 'sharpness', 'noise_level', 'edge_density']:
            if metric in original_metrics and metric in final_metrics:
                original_val = original_metrics[metric]
                final_val = final_metrics[metric]
                
                if metric == 'noise_level':
                    # For noise, reduction is improvement
                    improvement = original_val - final_val
                    improvement_pct = (improvement / original_val * 100) if original_val > 0 else 0
                else:
                    # For other metrics, increase is improvement
                    improvement = final_val - original_val
                    improvement_pct = (improvement / original_val * 100) if original_val > 0 else 0
                
                improvements[metric] = {
                    'absolute': float(improvement),
                    'percentage': float(improvement_pct),
                    'original': float(original_val),
                    'final': float(final_val)
                }
        
        return improvements
    
    def _validate_processing_quality(self, result: Dict) -> List[str]:
        """Validate processing quality and return warnings"""
        warnings = []
        
        original = result['original_metrics']
        final = result['final_metrics']
        
        # Check for over-processing
        if final['sharpness'] > original['sharpness'] * 3:
            warnings.append("Possible over-sharpening detected")
        
        # Check for excessive noise reduction
        if final['edge_density'] < original['edge_density'] * 0.5:
            warnings.append("Possible over-smoothing - important details may be lost")
        
        # Check for artifacts
        if final['noise_level'] > original['noise_level'] * 1.5:
            warnings.append("Processing may have introduced artifacts")
        
        # Check brightness adjustments
        brightness_change = abs(final['mean_brightness'] - original['mean_brightness'])
        if brightness_change > 100:
            warnings.append(f"Large brightness change detected ({brightness_change:.1f})")
        
        return warnings
    
    def _handle_processing_error(self, error: Exception, processing_time: float, 
                                source_path: str) -> Dict:
        """Handle processing errors and create error result"""
        error_msg = str(error)
        logger.error(f"Processing failed for {source_path}: {error_msg}")
        
        return {
            'source_path': source_path,
            'original_image': None,
            'processed_image': None,
            'intermediate_images': {},
            'processing_steps': [],
            'original_metrics': None,
            'final_metrics': None,
            'improvements': {},
            'processing_options': {},
            'processing_time': processing_time,
            'success': False,
            'error': error_msg,
            'warnings': [],
            'pipeline_version': '1.0.0'
        }
    
    def _update_processing_stats(self, processing_time: float, success: bool, error: str = None):
        """Update internal processing statistics"""
        self.stats['total_processing_time'] += processing_time
        
        if self.stats['total_processed'] > 0:
            self.stats['average_processing_time'] = (
                self.stats['total_processing_time'] / self.stats['total_processed']
            )
        
        if not success:
            self.stats['error_count'] += 1
            self.stats['last_error'] = error
    
    @timing_decorator
    def process_multiple_images(self, 
                              image_inputs: List[Union[np.ndarray, str, Path]], 
                              options: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Process multiple images with multi-angle support
        """
        if not image_inputs:
            raise ValueError("No images provided for processing")
        
        logger.info(f"Processing {len(image_inputs)} images")
        
        results = {
            'individual_results': [],
            'successful_count': 0,
            'failed_count': 0,
            'total_processing_time': 0.0,
            'composite_result': None,
            'best_result': None,
            'average_improvements': {}
        }
        
        # Process each image individually
        successful_results = []
        
        for idx, image_input in enumerate(image_inputs):
            logger.info(f"Processing image {idx + 1}/{len(image_inputs)}")
            
            try:
                result = self.process_single_image(image_input, options)
                results['individual_results'].append(result)
                results['total_processing_time'] += result['processing_time']
                
                if result['success']:
                    successful_results.append(result)
                    results['successful_count'] += 1
                else:
                    results['failed_count'] += 1
                    
            except Exception as e:
                logger.error(f"Failed to process image {idx + 1}: {str(e)}")
                results['failed_count'] += 1
        
        # Multi-angle processing if we have multiple successful results
        if len(successful_results) > 1:
            results['composite_result'] = self._create_multi_angle_composite(successful_results)
        
        # Find best individual result
        if successful_results:
            results['best_result'] = self._find_best_result(successful_results)
            results['average_improvements'] = self._calculate_average_improvements(successful_results)
        
        logger.info(f"Batch processing completed: {results['successful_count']} successful, "
                   f"{results['failed_count']} failed")
        
        return results
    
    def _create_multi_angle_composite(self, successful_results: List[Dict]) -> Dict:
        """Create composite from multiple angle processing"""
        logger.info("Creating multi-angle composite")
        
        # Simple approach: select the image with best overall quality
        best_result = max(successful_results, 
                         key=lambda x: x['final_metrics']['sharpness'] + 
                                     x['final_metrics']['contrast'] - 
                                     x['final_metrics']['noise_level'])
        
        composite_result = {
            'method': 'best_quality_selection',
            'selected_image': best_result['processed_image'],
            'source_result': best_result,
            'quality_score': (best_result['final_metrics']['sharpness'] + 
                            best_result['final_metrics']['contrast'] - 
                            best_result['final_metrics']['noise_level']),
            'total_candidates': len(successful_results)
        }
        
        return composite_result
    
    def _find_best_result(self, successful_results: List[Dict]) -> Dict:
        """Find the best individual processing result"""
        return max(successful_results, 
                  key=lambda x: x['final_metrics']['sharpness'])
    
    def _calculate_average_improvements(self, successful_results: List[Dict]) -> Dict:
        """Calculate average improvements across all successful results"""
        if not successful_results:
            return {}
        
        avg_improvements = {}
        metrics = ['sharpness', 'contrast', 'noise_level', 'mean_brightness']
        
        for metric in metrics:
            values = []
            for result in successful_results:
                if metric in result['improvements']:
                    values.append(result['improvements'][metric]['percentage'])
            
            if values:
                avg_improvements[metric] = {
                    'average_improvement_pct': sum(values) / len(values),
                    'min_improvement_pct': min(values),
                    'max_improvement_pct': max(values),
                    'sample_count': len(values)
                }
        
        return avg_improvements
    
    def get_processing_statistics(self) -> Dict:
        """Get comprehensive processing statistics"""
        stats = self.stats.copy()
        
        if stats['total_processed'] > 0:
            stats['success_rate'] = stats['successful_processed'] / stats['total_processed']
            stats['error_rate'] = stats['error_count'] / stats['total_processed']
        else:
            stats['success_rate'] = 0.0
            stats['error_rate'] = 0.0
        
        return stats
    
    def reset_statistics(self):
        """Reset processing statistics"""
        self.stats = {
            'total_processed': 0,
            'successful_processed': 0,
            'average_processing_time': 0.0,
            'total_processing_time': 0.0,
            'error_count': 0,
            'last_error': None
        }
        logger.info("Processing statistics reset")
    
    def benchmark_pipeline(self, test_images: List[np.ndarray], 
                          iterations: int = 3, options: Optional[Dict] = None) -> Dict:
        """Comprehensive pipeline benchmarking"""
        logger.info(f"Starting benchmark: {len(test_images)} images × {iterations} iterations")
        
        benchmark_start = time.time()
        all_results = []
        
        for iteration in range(iterations):
            logger.info(f"Benchmark iteration {iteration + 1}/{iterations}")
            
            for img_idx, image in enumerate(test_images):
                result = self.process_single_image(image, options)
                
                benchmark_record = {
                    'iteration': iteration,
                    'image_index': img_idx,
                    'processing_time': result['processing_time'],
                    'success': result['success'],
                    'steps_completed': len(result['processing_steps']),
                    'warnings_count': len(result['warnings']),
                    'final_metrics': result['final_metrics'] if result['success'] else None
                }
                
                all_results.append(benchmark_record)
        
        total_benchmark_time = time.time() - benchmark_start
        
        # Calculate benchmark statistics
        successful_results = [r for r in all_results if r['success']]
        processing_times = [r['processing_time'] for r in successful_results]
        
        benchmark_summary = {
            'total_benchmark_time': total_benchmark_time,
            'total_images_processed': len(all_results),
            'successful_processes': len(successful_results),
            'success_rate': len(successful_results) / len(all_results) if all_results else 0,
            'average_processing_time': sum(processing_times) / len(processing_times) if processing_times else 0,
            'min_processing_time': min(processing_times) if processing_times else 0,
            'max_processing_time': max(processing_times) if processing_times else 0,
            'median_processing_time': np.median(processing_times) if processing_times else 0,
            'processing_time_std': np.std(processing_times) if processing_times else 0,
            'detailed_results': all_results,
            'performance_target_met': all(t <= self.config.MAX_PROCESSING_TIME for t in processing_times)
        }
        
        logger.info(f"Benchmark completed in {total_benchmark_time:.2f}s")
        logger.info(f"Success rate: {benchmark_summary['success_rate']:.1%}")
        logger.info(f"Average processing time: {benchmark_summary['average_processing_time']:.3f}s")
        logger.info(f"Performance target met: {benchmark_summary['performance_target_met']}")
        
        return benchmark_summary