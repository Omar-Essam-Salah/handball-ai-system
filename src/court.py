import cv2
import numpy as np
import json
import os
from pathlib import Path
from ihf_standards import IHFHandballCourt

class CourtTransformer:
    def __init__(self, config_path="config/court_calibration.json"):
        self.homography_matrix = None
        self.standard_court = IHFHandballCourt()
        self.config_path = Path(config_path)
        self.load_calibration()

    def calibrate(self, pixel_points):
        """
        pixel_points: list of 4 tuples (x, y) representing the 4 corners of the court
        in this order: [Top-Left, Top-Right, Bottom-Right, Bottom-Left]
        """
        if len(pixel_points) != 4:
            return False

        src_pts = np.array(pixel_points, dtype=np.float32)
        # Corresponding IHF standard coordinates in meters
        dst_pts = np.array([
            [0, 0], 
            [self.standard_court.WIDTH, 0], 
            [self.standard_court.WIDTH, self.standard_court.HEIGHT], 
            [0, self.standard_court.HEIGHT]
        ], dtype=np.float32)

        self.homography_matrix, _ = cv2.findHomography(src_pts, dst_pts)
        self.save_calibration()
        return True

    def save_calibration(self):
        if self.homography_matrix is not None:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w") as f:
                json.dump({"homography": self.homography_matrix.tolist()}, f)

    def load_calibration(self):
        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    data = json.load(f)
                    if "homography" in data:
                        self.homography_matrix = np.array(data["homography"], dtype=np.float32)
            except Exception as e:
                print(f"[Court] Failed to load calibration: {e}")

    def is_calibrated(self):
        return self.homography_matrix is not None

    def pixel_to_meter(self, x, y):
        if not self.is_calibrated():
            return None
        
        pt = np.array([[[x, y]]], dtype=np.float32)
        transformed_pt = cv2.perspectiveTransform(pt, self.homography_matrix)
        return transformed_pt[0][0][0], transformed_pt[0][0][1] # Returns (x_meters, y_meters)

    def is_goal(self, x_m, y_m):
        """
        Check if the ball crossed the goal line.
        Goals are at x=0 and x=40. Width is 3m. So y is between 8.5m and 11.5m.
        """
        if y_m >= 8.5 and y_m <= 11.5:
            if x_m <= 0.0:
                return "left"
            elif x_m >= self.standard_court.WIDTH:
                return "right"
        return None
