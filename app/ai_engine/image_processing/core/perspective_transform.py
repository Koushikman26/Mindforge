import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
from ..utils.image_utils import timing_decorator, validate_image, logger
from ..configs.processing_config import ImageProcessingConfig

class PerspectiveTransformer:
    """Advanced perspective correction for ceiling-mounted cameras in retail environments"""
    
    def __init__(self, config: ImageProcessingConfig):
        self.config = config
        logger.info("Initialized PerspectiveTransformer")
    
    @timing_decorator
    def correct_ceiling_perspective(self, image: np.ndarray, shelf_detection: bool = None) -> np.ndarray:
        """
        Main perspective correction function optimized for ceiling cameras
        """
        if not validate_image(image):
            raise ValueError("Invalid image provided for perspective correction")
        
        if shelf_detection is None:
            shelf_detection = self.config.ENABLE_SHELF_DETECTION
        
        logger.info(f"Correcting perspective for ceiling camera, shelf_detection={shelf_detection}")
        
        if shelf_detection:
            # Try to detect shelf structure for guided correction
            if shelf_contours := self._detect_shelf_structure(image):
                return self._correct_using_shelf_geometry(image, shelf_contours)
        
        # Fallback to general document/rectangle detection
        document_corners = self._detect_rectangular_object(image)
        
        if document_corners is not None:
            return self._apply_perspective_correction(image, document_corners)
        else:
            # Last resort: apply standard ceiling camera correction
            return self._apply_ceiling_camera_correction(image)
    
    def _detect_shelf_structure(self, image: np.ndarray) -> List[np.ndarray]:
        """Detect shelf structures in the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Enhance edges for shelf detection
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 30, 100)
        
        # Detect horizontal lines (shelf edges)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
        horizontal_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        
        # Detect vertical lines (shelf supports)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
        vertical_lines = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
        
        # Combine to find shelf rectangles
        shelf_structure = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
        
        # Find contours representing shelf sections
        contours, _ = cv2.findContours(shelf_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size for shelf sections
        shelf_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > self.config.SHELF_DETECTION_MIN_AREA:
                # Approximate to rectangle
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                if len(approx) >= 4:  # Should be roughly rectangular
                    shelf_contours.append(approx)
        
        logger.info(f"Detected {len(shelf_contours)} shelf structures")
        return shelf_contours
    
    def _detect_rectangular_object(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Detect the largest rectangular object in the image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Multi-scale edge detection for robustness
        edges_list = []
        for sigma in [0.5, 1.0, 1.5]:
            blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
            edges = cv2.Canny(blurred, 40, 120)
            edges_list.append(edges)
        
        # Combine edge maps
        combined_edges = np.maximum.reduce(edges_list)
        
        # Morphological operations to close gaps
        kernel = np.ones((3, 3), np.uint8)
        combined_edges = cv2.morphologyEx(combined_edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(combined_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort by area and examine largest contours
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        
        for contour in contours[:15]:  # Check top 15 largest contours
            area = cv2.contourArea(contour)
            
            if area < self.config.MIN_CONTOUR_AREA:
                continue
            
            # Approximate contour to polygon
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Look for quadrilaterals
            if len(approx) == 4 and self._is_valid_rectangle(approx):
                corners = self._order_rectangle_points(approx.reshape(4, 2))
                logger.info(f"Found rectangular object with area {area}")
                return corners
        
        logger.info("No suitable rectangular object found")
        return None
    
    def _is_valid_rectangle(self, approx: np.ndarray) -> bool:
        """Validate that the approximated contour represents a reasonable rectangle"""
        if len(approx) != 4:
            return False
        
        # Calculate angles between consecutive edges
        points = approx.reshape(4, 2)
        angles = []
        
        for i in range(4):
            p1 = points[i]
            p2 = points[(i + 1) % 4]
            p3 = points[(i + 2) % 4]
            
            # Calculate vectors
            v1 = p2 - p1
            v2 = p3 - p2
            
            # Calculate angle
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle) * 180 / np.pi
            angles.append(angle)
        
        # Check if angles are close to 90 degrees (allowing some tolerance)
        angle_tolerance = 25  # degrees
        valid_angles = all(70 < angle < 110 for angle in angles)
        
        # Check aspect ratio (not too extreme)
        rect = cv2.boundingRect(approx)
        aspect_ratio = max(rect[2], rect[3]) / min(rect[2], rect[3])
        reasonable_aspect = aspect_ratio < 5.0
        
        return valid_angles and reasonable_aspect
    
    def _order_rectangle_points(self, pts: np.ndarray) -> np.ndarray:
        """Order rectangle points in consistent order: top-left, top-right, bottom-right, bottom-left"""
        rect = np.zeros((4, 2), dtype=np.float32)
        
        # Sum and difference of coordinates
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        # Top-left point has smallest sum
        rect[0] = pts[np.argmin(s)]
        
        # Bottom-right point has largest sum
        rect[2] = pts[np.argmax(s)]
        
        # Top-right point has smallest difference (x - y)
        rect[1] = pts[np.argmin(diff)]
        
        # Bottom-left point has largest difference
        rect[3] = pts[np.argmax(diff)]
        
        return rect
    
    def _apply_perspective_correction(self, image: np.ndarray, corners: np.ndarray) -> np.ndarray:
        """Apply perspective correction using detected corners"""
        width, height = self._calculate_dimensions(corners)
        
        # Ensure minimum dimensions
        width = max(width, 400)
        height = max(height, 300)
        
        # Define destination rectangle
        dst_points = np.array([
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ], dtype=np.float32)
        
        # Calculate perspective transformation matrix
        matrix = cv2.getPerspectiveTransform(corners, dst_points)
        
        # Apply transformation and return immediately
        logger.info(f"Applied perspective correction, output size: {width}x{height}")
        return cv2.warpPerspective(image, matrix, (width, height))

    def _calculate_dimensions(self, corners: np.ndarray) -> Tuple[int, int]:
        """Calculate width and height for perspective correction"""
        width = max(
            int(self._distance(corners[1], corners[0])),
            int(self._distance(corners[2], corners[3]))
        )
        height = max(
            int(self._distance(corners[3], corners[0])),
            int(self._distance(corners[2], corners[1]))
        )
        return width, height

    def _distance(self, pt1: np.ndarray, pt2: np.ndarray) -> float:
        """Calculate Euclidean distance between two points"""
        return np.linalg.norm(pt1 - pt2)
    
    def _apply_ceiling_camera_correction(self, image: np.ndarray) -> np.ndarray:
        """Apply standard correction for ceiling-mounted cameras"""
        h, w = image.shape[:2]
        
        # Define typical perspective distortion for ceiling cameras
        # Assume the camera is looking down at an angle
        src_points = np.array([
            [w * 0.15, h * 0.15],  # Top-left
            [w * 0.85, h * 0.15],  # Top-right
            [w * 0.95, h * 0.85],  # Bottom-right
            [w * 0.05, h * 0.85]   # Bottom-left
        ], dtype=np.float32)
        
        dst_points = np.array([
            [0, 0],
            [w, 0],
            [w, h],
            [0, h]
        ], dtype=np.float32)
        
        # Calculate and apply transformation
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        corrected = cv2.warpPerspective(image, matrix, (w, h))
        
        logger.info("Applied standard ceiling camera perspective correction")
        return corrected
    
    def _correct_using_shelf_geometry(self, image: np.ndarray, shelf_contours: List[np.ndarray]) -> np.ndarray:
        """Correct perspective using detected shelf geometry"""
        # Find the largest shelf section to use as reference
        largest_shelf = max(shelf_contours, key=cv2.contourArea)
        
        # Approximate to rectangle
        epsilon = 0.02 * cv2.arcLength(largest_shelf, True)
        approx = cv2.approxPolyDP(largest_shelf, epsilon, True)
        
        if len(approx) >= 4:
            # Use the shelf rectangle for perspective correction
            corners = self._order_rectangle_points(approx[:4].reshape(4, 2))
            return self._apply_perspective_correction(image, corners)
        # Fallback to standard correction
        return self._apply_ceiling_camera_correction(image)
    
    @timing_decorator
    def correct_rotation(self, image: np.ndarray) -> np.ndarray:
        """Detect and correct image rotation"""
        if not validate_image(image):
            raise ValueError("Invalid image provided for rotation correction")
        
        angle = self._detect_rotation_angle(image)
        
        if abs(angle) > self.config.ROTATION_TOLERANCE:
            logger.info(f"Correcting rotation by {angle:.2f} degrees")
            return self._rotate_image(image, -angle)
        else:
            logger.info(f"No significant rotation detected (angle: {angle:.2f}°)")
            return image
    
    def _detect_rotation_angle(self, image: np.ndarray) -> float:
        """Detect rotation angle using line detection"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # Detect lines using HoughLines
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=80)
        
        if lines is not None:
            angles = []
            
            for rho, theta in lines[:30]:  # Use top 30 lines for better statistics
                angle = theta * 180 / np.pi
                
                # Convert to -90 to 90 range
                if angle > 90:
                    angle -= 180
                elif angle < -90:
                    angle += 180
                
                # Focus on nearly horizontal lines (for shelf edges)
                if abs(angle) < 45:
                    angles.append(angle)
            
            if angles:
                # Use median for robustness against outliers
                median_angle = np.median(angles)
                logger.info(f"Detected rotation angle: {median_angle:.2f}° from {len(angles)} lines")
                return median_angle
        
        logger.info("No clear rotation angle detected")
        return 0.0
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image by specified angle with proper bounds calculation"""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Calculate rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Calculate new image dimensions to fit rotated image
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Adjust rotation matrix for new center
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]
        
        # Apply rotation
        return cv2.warpAffine(image, rotation_matrix, (new_w, new_h), 
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
    def auto_crop_rotated_image(self, image: np.ndarray) -> np.ndarray:
        """Automatically crop rotated image to remove black borders"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find non-zero regions
        coords = np.column_stack(np.where(gray > 0))
        
        if len(coords) > 0:
            # Get bounding rectangle
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            
            # Crop image
            cropped = image[y_min:y_max+1, x_min:x_max+1]
            logger.info(f"Auto-cropped image from {image.shape[:2]} to {cropped.shape[:2]}")
            return cropped
        
        return image