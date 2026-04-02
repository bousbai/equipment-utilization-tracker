import cv2
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ActivityClassifier:
    """Classifies equipment activities based on motion patterns and geometry."""
    
    # Activity types
    DIGGING = 'Digging'
    SWINGING = 'Swinging/Loading'
    DUMPING = 'Dumping'
    WAITING = 'Waiting'
    
    def __init__(self, history_frames: int = 30):
        """Initialize activity classifier."""
        self.history_frames = history_frames
        self.activity_history = defaultdict(list)
        self.bbox_history = defaultdict(list)
        self.motion_history = defaultdict(list)
        
    def classify(self, motion_data: Dict, equipment_data: Dict, 
                 frame: np.ndarray) -> Dict:
        """
        Classify activities for each detected equipment.
        
        Args:
            motion_data: Motion analysis results
            equipment_data: Equipment detection data
            frame: Current video frame
            
        Returns:
            Dictionary with activity classification for each equipment
        """
        activity_results = {}
        
        for eq_id, motion_info in motion_data.items():
            equipment_id = motion_info['equipment_id']
            equipment_type = motion_info['type']
            is_active = motion_info['is_active']
            bbox = motion_info['bbox']
            motion_pct = motion_info['motion_percentage']
            
            # Update history
            self.bbox_history[eq_id].append(bbox)
            self.motion_history[eq_id].append(motion_pct)
            
            if len(self.bbox_history[eq_id]) > self.history_frames:
                self.bbox_history[eq_id].pop(0)
                self.motion_history[eq_id].pop(0)
            
            # Determine activity
            if not is_active:
                activity = self.WAITING
            else:
                activity = self._classify_activity(
                    eq_id, equipment_type, bbox, frame, motion_pct
                )
            
            # Update activity history for smoothing
            self.activity_history[eq_id].append(activity)
            if len(self.activity_history[eq_id]) > self.history_frames:
                self.activity_history[eq_id].pop(0)
            
            # Use most common activity in recent history
            smoothed_activity = max(set(self.activity_history[eq_id]), 
                                   key=self.activity_history[eq_id].count)
            
            activity_results[eq_id] = {
                'equipment_id': equipment_id,
                'type': equipment_type,
                'activity': smoothed_activity,
                'confidence': self._get_activity_confidence(eq_id),
                'bbox': bbox,
                'is_active': is_active,
                'motion_percentage': motion_pct
            }
        
        return activity_results
    
    def _classify_activity(self, eq_id: int, equipment_type: str, 
                          bbox: Tuple[int, int, int, int], 
                          frame: np.ndarray, motion_pct: float) -> str:
        """
        Classify specific activity for equipment.
        
        Activity heuristics:
        - Digging: Excavator with downward arm motion + high motion in lower region
        - Swinging: High lateral motion + rotating motion pattern
        - Dumping: Truck or loader with raised bucket/bed + motion
        - Waiting: Low motion, stationary
        """
        
        if equipment_type == 'excavator':
            return self._classify_excavator_activity(eq_id, bbox, frame, motion_pct)
        elif equipment_type in ['dump_truck', 'loader']:
            return self._classify_truck_activity(eq_id, bbox, frame, motion_pct)
        elif equipment_type == 'bulldozer':
            return self._classify_bulldozer_activity(eq_id, bbox, frame, motion_pct)
        else:
            return self.WAITING
    
    def _classify_excavator_activity(self, eq_id: int, bbox: Tuple, 
                                     frame: np.ndarray, motion_pct: float) -> str:
        """Classify excavator-specific activities."""
        
        x1, y1, x2, y2 = bbox
        height = y2 - y1
        width = x2 - x1
        
        # Analyze motion distribution
        motion_hist = self.motion_history[eq_id]
        if not motion_hist:
            return self.WAITING
        
        # High sustained motion suggests active digging
        avg_motion = np.mean(motion_hist[-10:]) if len(motion_hist) >= 10 else motion_pct
        
        # Check for vertical motion pattern (digging motion)
        if avg_motion > 0.05:  # Significant motion threshold
            # Excavators typically have rotational motion when swinging
            if len(motion_hist) >= 5:
                motion_trend = motion_hist[-5:]
                
                # Check if motion is consistent (digging) vs oscillating (swinging)
                motion_var = np.var(motion_trend)
                
                if motion_var < 0.01 and avg_motion > 0.08:
                    # Consistent high motion = digging
                    return self.DIGGING
                elif motion_var > 0.015:
                    # Variable motion pattern = swinging/loading
                    return self.SWINGING
        
        return self.WAITING
    
    def _classify_truck_activity(self, eq_id: int, bbox: Tuple,
                                frame: np.ndarray, motion_pct: float) -> str:
        """Classify truck/loader activities."""
        
        x1, y1, x2, y2 = bbox
        
        motion_hist = self.motion_history[eq_id]
        if not motion_hist:
            return self.WAITING
        
        avg_motion = np.mean(motion_hist[-10:]) if len(motion_hist) >= 10 else motion_pct
        
        # Check for dumping (vertical motion in upper region of bbox, indicating raised bed)
        roi = frame[y1:y2, x1:x2]
        
        if avg_motion > 0.04:
            # Analyze motion distribution (raised bed/bucket creates upward motion)
            try:
                roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
                # Motion in upper third suggests dumping
                upper_third = roi[:roi.shape[0]//3, :]
                lower_third = roi[2*roi.shape[0]//3:, :]
                
                # If more variance in upper region, likely dumping
                upper_var = np.var(upper_third)
                lower_var = np.var(lower_third)
                
                if upper_var > lower_var * 1.2:
                    return self.DUMPING
            except:
                pass
            
            return self.SWINGING
        
        return self.WAITING
    
    def _classify_bulldozer_activity(self, eq_id: int, bbox: Tuple,
                                     frame: np.ndarray, motion_pct: float) -> str:
        """Classify bulldozer activities."""
        
        motion_hist = self.motion_history[eq_id]
        if not motion_hist:
            return self.WAITING
        
        avg_motion = np.mean(motion_hist[-10:]) if len(motion_hist) >= 10 else motion_pct
        
        if avg_motion > 0.06:
            return self.DIGGING  # Bulldozers primarily dig/push
        
        return self.WAITING
    
    def _get_activity_confidence(self, eq_id: int) -> float:
        """Get confidence score for current activity classification."""
        
        if eq_id not in self.activity_history:
            return 0.5
        
        activity_hist = self.activity_history[eq_id]
        if not activity_hist:
            return 0.5
        
        # More consistency = higher confidence
        current_activity = activity_hist[-1]
        count = activity_hist.count(current_activity)
        confidence = min(1.0, count / self.history_frames)
        
        return float(confidence)