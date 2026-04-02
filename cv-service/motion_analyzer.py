import cv2
import numpy as np
from typing import Dict, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MotionAnalyzer:
    """Analyzes motion in equipment regions for ACTIVE/INACTIVE state detection."""
    
    def __init__(self, motion_threshold: float = 0.02, history_frames: int = 30):
        """
        Initialize motion analyzer.
        
        Args:
            motion_threshold: Percentage of pixels with motion to trigger ACTIVE
            history_frames: Number of frames to maintain history for smoothing
        """
        self.motion_threshold = motion_threshold
        self.history_frames = history_frames
        self.frame_history = {}
        self.motion_history = {}
        self.prev_gray = None
        
    def analyze(self, frame: np.ndarray, equipment_data: Dict) -> Dict:
        """
        Analyze motion for each equipment.
        
        Args:
            frame: Current frame (BGR)
            equipment_data: Dictionary of equipment with bounding boxes
            
        Returns:
            Dictionary with motion analysis results and ACTIVE/INACTIVE states
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        
        motion_results = {}
        
        for eq_id, eq_info in equipment_data.items():
            bbox = eq_info['bbox']
            equipment_type = eq_info['type']
            
            # Extract region of interest
            x1, y1, x2, y2 = bbox
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(frame.shape[1], x2)
            y2 = min(frame.shape[0], y2)
            
            roi_gray = gray[y1:y2, x1:x2]
            
            if self.prev_gray is None or roi_gray.size == 0:
                motion_pct = 0.0
                is_active = False
            else:
                # Get motion in this region
                motion_pct = self._calculate_motion(roi_gray, x1, y1, frame.shape[:2])
                
                # Update motion history
                if eq_id not in self.motion_history:
                    self.motion_history[eq_id] = []
                
                self.motion_history[eq_id].append(motion_pct)
                
                # Keep only recent history
                if len(self.motion_history[eq_id]) > self.history_frames:
                    self.motion_history[eq_id].pop(0)
                
                # Use median to smooth noise
                motion_history = self.motion_history[eq_id]
                smoothed_motion = np.median(motion_history) if motion_history else 0
                
                # Determine ACTIVE/INACTIVE with hysteresis
                is_active = smoothed_motion > self.motion_threshold
            
            motion_results[eq_id] = {
                'equipment_id': eq_info.get('equipment_id', f'eq_{eq_id}'),
                'type': equipment_type,
                'motion_percentage': float(motion_pct),
                'is_active': is_active,
                'status': 'ACTIVE' if is_active else 'INACTIVE',
                'bbox': bbox,
                'center': eq_info['center'],
                'confidence': eq_info.get('confidence', 0.0)
            }
        
        self.prev_gray = gray
        return motion_results
    
    def _calculate_motion(self, roi_gray: np.ndarray, x_offset: int, 
                         y_offset: int, frame_shape: Tuple) -> float:
        """
        Calculate optical flow-based motion percentage in ROI.
        
        Args:
            roi_gray: Grayscale ROI
            x_offset, y_offset: ROI position in frame
            frame_shape: Original frame dimensions
            
        Returns:
            Percentage of pixels with significant motion
        """
        if self.prev_gray is None or roi_gray.size == 0:
            return 0.0
        
        try:
            # Extract corresponding region from previous frame
            y1, y2 = y_offset, min(y_offset + roi_gray.shape[0], frame_shape[0])
            x1, x2 = x_offset, min(x_offset + roi_gray.shape[1], frame_shape[1])
            
            prev_roi = self.prev_gray[y1:y2, x1:x2]
            
            if prev_roi.shape != roi_gray.shape or prev_roi.size == 0:
                return 0.0
            
            # Calculate dense optical flow (Farneback)
            flow = cv2.calcOpticalFlowFarneback(
                prev_roi, roi_gray,
                None,  # flow
                0.5,   # pyr_scale
                3,     # levels
                15,    # winsize
                3,     # iterations
                5,     # poly_n
                1.2,   # poly_sigma
                0      # flags
            )
            
            # Calculate magnitude of optical flow
            mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            
            # Motion threshold (pixels per frame)
            motion_threshold = 2.0
            motion_mask = mag > motion_threshold
            
            motion_percentage = np.sum(motion_mask) / motion_mask.size
            
            return float(motion_percentage)
        
        except Exception as e:
            logger.warning(f"Motion calculation error: {e}")
            return 0.0
    
    def reset(self):
        """Reset motion history (for new video or sequence)."""
        self.prev_gray = None
        self.motion_history.clear()