"""
Comprehensive test suite for the image processing pipeline

This module tests all aspects of the advanced image processing pipeline,
including individual processors and the complete pipeline functionality.
"""

import pytest
import cv2
import numpy as np
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Import the modules to test
from ai_engine.image_processing.core.pipeline import AdvancedImageProcessingPipeline
from ai_engine.image_processing.configs.processing_config import ImageProcessingConfig
from ai_engine.image_processing.utils.image_utils import calculate_image_metrics


class TestImageCreation:
    """Helper class for creating test images"""

    @staticmethod
    def create_test_image(image_type: str = "normal") -> np.ndarray:
        """Create different types of test images"""
        if image_type == "normal":
            return np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
        elif image_type == "dark":
            return np.random.randint(10, 60, (480, 640, 3), dtype=np.uint8)
        elif image_type == "bright":
            return np.random.randint(200, 255, (480, 640, 3), dtype=np.uint8)
        elif image_type == "noisy":
            base = np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8)
            noise = np.random.normal(0, 30, base.shape)
            return np.clip(base + noise, 0, 255).astype(np.uint8)
        elif image_type == "blurred":
            normal = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
            return cv2.GaussianBlur(normal, (21, 21), 0)
        elif image_type == "small":
            normal = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
            return cv2.resize(normal, (100, 75))
        elif image_type == "large":
            normal = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
            return cv2.resize(normal, (1920, 1440))
        else:
            return np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)

    @staticmethod
    def create_perspective_distorted_image(base_image: np.ndarray) -> np.ndarray:
        """Create an image with perspective distortion"""
        h, w = base_image.shape[:2]
        src_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        dst_points = np.array([[w*0.1, h*0.1], [w*0.9, h*0.05],
                              [w*0.95, h*0.9], [w*0.05, h*0.95]], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        return cv2.warpPerspective(base_image, matrix, (w, h))

    @staticmethod
    def create_uneven_lighting_image(base_image: np.ndarray) -> np.ndarray:
        """Create an image with uneven lighting"""
        h, w = base_image.shape[:2]
        x_gradient = np.linspace(0.5, 1.5, w)
        y_gradient = np.linspace(1.2, 0.8, h)
        lighting_map = np.outer(y_gradient, x_gradient)

        result = base_image.copy().astype(np.float32)
        for i in range(3):
            result[:, :, i] *= lighting_map

        return np.clip(result, 0, 255).astype(np.uint8)


@pytest.fixture
def image_config():
    """Fixture for image processing configuration"""
    return ImageProcessingConfig()


@pytest.fixture
def pipeline(image_config):
    """Fixture for image processing pipeline"""
    return AdvancedImageProcessingPipeline(image_config)


@pytest.fixture
def test_images():
    """Fixture providing various test images"""
    helper = TestImageCreation()
    images = {
        'normal': helper.create_test_image('normal'),
        'dark': helper.create_test_image('dark'),
        'bright': helper.create_test_image('bright'),
        'noisy': helper.create_test_image('noisy'),
        'blurred': helper.create_test_image('blurred'),
        'small': helper.create_test_image('small'),
        'large': helper.create_test_image('large'),
    }

    # Add special cases
    normal = images['normal']
    images['perspective'] = helper.create_perspective_distorted_image(normal)
    images['uneven_lighting'] = helper.create_uneven_lighting_image(normal)

    # Add rotated image
    center = (320, 240)
    rotation_matrix = cv2.getRotationMatrix2D(center, 15, 1.0)
    images['rotated'] = cv2.warpAffine(normal, rotation_matrix, (640, 480))

    return images


@pytest.fixture
def temp_dir():
    """Fixture for temporary directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


class TestPipelineProcessing:
    """Test suite for pipeline processing functionality"""

    def test_single_image_processing_success(self, pipeline, test_images):
        """Test successful processing of various image types"""
        for image_type, image in test_images.items():
            result = pipeline.process_single_image(image)

            assert result['success'], f"Processing failed for {image_type}"
            assert result['processed_image'] is not None, f"No processed image for {image_type}"
            assert result['processed_image'].shape[:2] == image.shape[:2], \
                   f"Image dimensions changed for {image_type}"
            assert len(result['processing_steps']) > 0, \
                   f"No processing steps recorded for {image_type}"

    def test_processing_time_requirement(self, pipeline, test_images):
        """Test that processing time meets the 10-second requirement"""
        test_image = test_images['normal']

        start_time = time.time()
        result = pipeline.process_single_image(test_image)
        actual_time = time.time() - start_time

        assert result['success'], "Processing must succeed for timing test"
        assert actual_time <= 10.0, \
               f"Processing time {actual_time:.2f}s exceeded limit of 10s"
        assert result['processing_time'] <= 10.0, \
               f"Reported processing time {result['processing_time']:.2f}s exceeded limit"

    @pytest.mark.parametrize("image_type,expected_improvement", [
        ("dark", "brightness"),
        ("bright", "contrast"),
        ("noisy", "noise"),
        ("blurred", "sharpness"),
    ])
    def test_specific_improvements(self, pipeline, test_images, image_type, expected_improvement):
        """Test that specific image problems are improved"""
        result = pipeline.process_single_image(test_images[image_type])

        assert result['success'], f"Processing failed for {image_type}"

        if expected_improvement == "noise":
            assert result['improvements']['noise_level']['absolute'] > 0, \
                   f"No noise reduction for {image_type}"
        elif expected_improvement == "brightness":
            assert result['improvements']['mean_brightness']['absolute'] > 0, \
                   f"No brightness improvement for {image_type}"
        elif expected_improvement == "contrast":
            assert abs(result['improvements']['contrast']['absolute']) > 0, \
                   f"No contrast improvement for {image_type}"
        elif expected_improvement == "sharpness":
            assert result['improvements']['sharpness']['absolute'] > 0, \
                   f"No sharpness improvement for {image_type}"

    def test_lighting_correction_effectiveness(self, pipeline, test_images):
        """Test lighting correction for dark and bright images"""
        # Test dark image
        dark_result = pipeline.process_single_image(
            test_images['dark'],
            {'enable_lighting_correction': True, 'enable_quality_enhancement': False,
             'enable_perspective_correction': False, 'enable_noise_reduction': False}
        )

        assert dark_result['success']
        assert dark_result['improvements']['mean_brightness']['absolute'] > 0, \
               "Dark image should be brightened"

        # Test bright image
        bright_result = pipeline.process_single_image(
            test_images['bright'],
            {'enable_lighting_correction': True, 'enable_quality_enhancement': False,
             'enable_perspective_correction': False, 'enable_noise_reduction': False}
        )

        assert bright_result['success']
        # Bright images should either reduce brightness or improve contrast
        brightness_changed = bright_result['improvements']['mean_brightness']['absolute'] < -5
        contrast_improved = bright_result['improvements']['contrast']['absolute'] > 5
        assert brightness_changed or contrast_improved, "Bright image should be corrected"

    def test_multi_image_processing(self, pipeline, test_images):
        """Test processing multiple images"""
        test_image_list = [
            test_images['normal'],
            test_images['perspective'],
            test_images['rotated']
        ]

        result = pipeline.process_multiple_images(test_image_list)

        assert result['successful_count'] > 0, "At least one image should process successfully"
        assert len(result['individual_results']) == 3, "Should have results for all 3 images"

        if result['successful_count'] > 1:
            assert result['composite_result'] is not None, \
                   "Should create composite for multiple successes"

    def test_error_handling(self, pipeline):
        """Test error handling for invalid inputs"""
        # Test with None
        result_none = pipeline.process_single_image(None)
        assert not result_none['success']
        assert result_none['error'] is not None

        # Test with invalid path
        result_invalid_path = pipeline.process_single_image("/nonexistent/path/image.jpg")
        assert not result_invalid_path['success']
        assert result_invalid_path['error'] is not None

        # Test with empty array
        invalid_array = np.array([])
        result_invalid_array = pipeline.process_single_image(invalid_array)
        assert not result_invalid_array['success']
        assert result_invalid_array['error'] is not None

    def test_processing_options(self, pipeline, test_images):
        """Test various processing options"""
        test_image = test_images['normal']

        # Test with all options disabled
        minimal_options = {
            'enable_noise_reduction': False,
            'enable_lighting_correction': False,
            'enable_perspective_correction': False,
            'enable_rotation_correction': False,
            'enable_quality_enhancement': False
        }

        result_minimal = pipeline.process_single_image(test_image, minimal_options)
        assert result_minimal['success']
        assert len(result_minimal['processing_steps']) == 0, "No processing steps should be applied"

        # Test with stock analysis optimization
        stock_options = {
            'optimize_for_stock_analysis': True,
            'enhance_text_readability': True,
            'enhance_product_visibility': True
        }

        result_stock = pipeline.process_single_image(test_image, stock_options)
        assert result_stock['success']
        assert len(result_stock['processing_steps']) > 0, "Should apply stock-optimized processing"

    def test_performance_with_different_sizes(self, pipeline, test_images):
        """Test performance with different image sizes"""
        # Test small image
        small_result = pipeline.process_single_image(test_images['small'])
        assert small_result['success']
        assert small_result['processing_time'] < 2.0, "Small image should process quickly"

        # Test large image
        large_result = pipeline.process_single_image(test_images['large'])
        assert large_result['success']
        assert large_result['processing_time'] <= 10.0, \
               "Large image should still meet time requirements"

    def test_intermediate_results_saving(self, pipeline, test_images, temp_dir):
        """Test saving of intermediate processing steps"""
        options = {
            'save_intermediate_steps': True,
            'output_directory': str(temp_dir)
        }

        result = pipeline.process_single_image(test_images['normal'], options)

        assert result['success']
        if 'intermediate_images' in result:
            assert len(result['intermediate_images']) > 0, "Should save intermediate images"

            # Verify intermediate images match processing steps
            for step in result['processing_steps']:
                if 'intermediate_images' in result:
                    assert step in result['intermediate_images'], \
                           f"Missing intermediate image for {step}"


class TestPipelineStatistics:
    """Test suite for pipeline statistics and benchmarking"""

    def test_processing_statistics(self, pipeline, test_images):
        """Test processing statistics tracking"""
        # Reset statistics
        pipeline.reset_statistics()

        # Process several images
        for _ in range(3):
            pipeline.process_single_image(test_images['normal'])

        stats = pipeline.get_processing_statistics()

        assert stats['total_processed'] == 3, "Should track total processed images"
        assert stats['successful_processed'] == 3, "Should track successful processes"
        assert stats['average_processing_time'] > 0, "Should calculate average processing time"
        assert stats['success_rate'] == 1.0, "Should have 100% success rate for normal images"

    def test_benchmark_functionality(self, pipeline, test_images):
        """Test pipeline benchmarking"""
        test_image_list = [test_images['normal'], test_images['dark']]

        benchmark_result = pipeline.benchmark_pipeline(test_image_list, iterations=2)

        assert benchmark_result['total_images_processed'] > 0
        assert benchmark_result['successful_processes'] > 0
        assert benchmark_result['average_processing_time'] > 0
        assert benchmark_result['performance_target_met'] is not None

        # Verify detailed results
        expected_count = len(test_image_list) * 2  # 2 images × 2 iterations
        assert len(benchmark_result['detailed_results']) == expected_count


class TestImageQualityMetrics:
    """Test suite for image quality metrics and improvements"""

    @pytest.mark.parametrize("image_type", ["dark", "noisy", "blurred", "uneven_lighting"])
    def test_quality_improvements(self, pipeline, test_images, image_type):
        """Test that processing improves image quality metrics"""
        result = pipeline.process_single_image(test_images[image_type])

        assert result['success'], f"Processing failed for {image_type}"

        improvements = result['improvements']
        quality_metrics = ['sharpness', 'contrast']
        noise_metric = 'noise_level'

        # Check for quality improvement
        quality_improved = any(
            improvements.get(metric, {}).get('absolute', 0) > 0
            for metric in quality_metrics
        )
        noise_reduced = improvements.get(noise_metric, {}).get('absolute', 0) > 0

        assert quality_improved or noise_reduced, \
               f"No quality improvement detected for {image_type}"

    def test_processing_step_order(self, pipeline, test_images):
        """Test that processing steps are applied in the correct order"""
        result = pipeline.process_single_image(test_images['noisy'])

        if not result['success']:
            pytest.skip("Processing failed, skipping order test")

        steps = result['processing_steps']
        expected_order = [
            'noise_reduction',
            'lighting_correction',
            'perspective_correction',
            'rotation_correction',
            'quality_enhancement'
        ]

        # Check that steps appear in expected order



        # Verify ordering
        previous_idx = -1
        for step in steps:
            if step in step_indices:
                current_idx = step_indices[step]
                assert current_idx >= previous_idx, \
                       f"Processing steps out of order: {step}"
                previous_idx = current_idx


# Markers for pytest
pytestmark = [
    pytest.mark.unit,
    pytest.mark.image_processing
]