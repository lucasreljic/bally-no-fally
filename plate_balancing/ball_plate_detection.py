"""Ball Detection Module for Ball and Beam System.

Computer vision functions for detecting and tracking ball position
using HSV color filtering and contour analysis.
"""

# Ball Detection Module for Computer Vision Ball Tracking System
# Detects colored balls in video frames using HSV color space filtering
# Provides both class-based and legacy function interfaces

import json
import os
import time

import cv2
import numpy as np


# --- new: simple EKF/KF for 3D position + velocity ---
class ExtendedKalmanFilter:
    """Simple (E)KF implementation with state [x,y,z,vx,vy,vz].

    Although the measurement model is linear (position), the class is named EKF
    so it can be extended to nonlinear measurements later (e.g. pixel->3D).
    """

    def __init__(self):
        """Initialize EKF with default parameters."""
        # State vector (6x1): x, y, z, vx, vy, vz
        self.x = np.zeros((6, 1), dtype=float)
        # Covariance
        self.P = np.eye(6, dtype=float) * 1.0
        # Process noise (tune as needed)
        self.Q = np.diag([0.01, 0.01, 0.02, 0.5, 0.5, 0.5])
        # Measurement noise (position measurements)
        self.R = np.diag([0.02, 0.02, 0.05])
        # Measurement matrix (maps state to measured position)
        self.H = np.zeros((3, 6), dtype=float)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self._initialized = False

    def initialize(self, pos3):
        """Initialize state position; velocity initialized to zero."""
        self.x[:3, 0] = np.array(pos3).reshape(3)
        self.x[3:, 0] = 0.0
        self.P = np.eye(6, dtype=float) * 0.5
        self._initialized = True

    def predict(self, dt):
        """Predict step with constant velocity model."""
        if not self._initialized:
            return

        if dt <= 0:
            return

        matrix_F = np.eye(6, dtype=float)
        matrix_F[0, 3] = dt
        matrix_F[1, 4] = dt
        matrix_F[2, 5] = dt

        # Simple discrete-time propagation
        self.x = matrix_F.dot(self.x)
        self.P = matrix_F.dot(self.P).dot(matrix_F.T) + self.Q

    def update(self, z):
        """Linear KF update with position measurement z (3-vector)."""
        if z is None:
            return

        z = np.asarray(z).reshape(3, 1)
        if not self._initialized:
            self.initialize(z.flatten())
            return

        matrix_h = self.H
        y = z - matrix_h.dot(self.x)  # innovation
        matrix_s = matrix_h.dot(self.P).dot(matrix_h.T) + self.R
        matrix_k = self.P.dot(matrix_h.T).dot(np.linalg.inv(matrix_s))
        self.x = self.x + matrix_k.dot(y)
        matrix_i = np.eye(self.P.shape[0])
        self.P = (matrix_i - matrix_k.dot(matrix_h)).dot(self.P)

    def get_state(self):
        """Return position and velocity as tuples: (pos3_tuple, vel3_tuple)."""
        pos = tuple(self.x[:3, 0].tolist())
        vel = tuple(self.x[3:, 0].tolist())
        return pos, vel


class BallDetector:
    """Computer vision ball detector using HSV color space filtering."""

    def __init__(
        self, config_file="ball_config.json", calibration_file="camera_calibration.npz"
    ):
        """Initialize ball detector with HSV bounds from config file.

        Args:
            config_file (str): Path to JSON config file with HSV bounds and calibration
            calibration_file (str): Path to camera calibration .npz file
        """
        # Default HSV bounds for orange ball detection
        self.lower_hsv = np.array([5, 150, 150], dtype=np.uint8)  # Orange lower bound
        self.upper_hsv = np.array([20, 255, 255], dtype=np.uint8)  # Orange upper bound
        self.scale_factor = 1.0  # Conversion factor from normalized coords to meters
        self.ball_diameter_m = 0.04  # Default ball diameter in meters (40mm)

        # Camera calibration parameters
        self.camera_matrix = None
        self.dist_coeffs = None
        self.fx = self.fy = self.cx = self.cy = None

        # Default min/max radii in pixels (used if config not provided)
        self.min_radius = 5
        self.max_radius = 1000

        # Extended Kalman Filter for 3D tracking
        self.ekf = ExtendedKalmanFilter()
        # Timekeeping for predict step
        self.last_time = time.time()

        # Future prediction visualization: times (s) into the future to show
        self.future_time_steps = [0.05, 0.1, 0.15]

        # Load camera calibration
        self._load_camera_calibration(calibration_file)

        # Load configuration from file if it exists
        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    config = json.load(f)

                # Extract HSV color bounds from config
                if "ball_detection" in config:
                    if config["ball_detection"]["lower_hsv"]:
                        self.lower_hsv = np.array(
                            config["ball_detection"]["lower_hsv"], dtype=np.uint8
                        )
                    if config["ball_detection"]["upper_hsv"]:
                        self.upper_hsv = np.array(
                            config["ball_detection"]["upper_hsv"], dtype=np.uint8
                        )
                    if "ball_diameter_m" in config["ball_detection"]:
                        self.ball_diameter_m = config["ball_detection"][
                            "ball_diameter_m"
                        ]
                    if "min_radius" in config["ball_detection"]:
                        self.min_radius = config["ball_detection"]["min_radius"]
                    if "max_radius" in config["ball_detection"]:
                        self.max_radius = config["ball_detection"]["max_radius"]

                print(
                    f"[BALL_DETECT] Loaded HSV bounds: {self.lower_hsv} to {self.upper_hsv}"
                )
                print(f"[BALL_DETECT] Ball diameter: {self.ball_diameter_m:.3f}m")

            except Exception as e:
                print(f"[BALL_DETECT] Config load error: {e}, using defaults")
        else:
            print("[BALL_DETECT] No config file found, using default HSV bounds")

    def _load_camera_calibration(self, calibration_file):
        """Load camera intrinsic parameters from .npz calibration file.

        Args:
            calibration_file (str): Path to .npz file with camera_matrix and dist_coeffs
        """
        try:
            data = np.load(calibration_file)
            self.camera_matrix = np.asarray(data["camera_matrix"], float).reshape(3, 3)
            self.dist_coeffs = np.asarray(data["dist_coeffs"], float).ravel()

            # Extract intrinsic parameters
            self.fx, self.fy = self.camera_matrix[0, 0], self.camera_matrix[1, 1]
            self.cx, self.cy = self.camera_matrix[0, 2], self.camera_matrix[1, 2]

            print(f"[BALL_DETECT] Loaded camera calibration from {calibration_file}")
            print(
                f"[BALL_DETECT] fx={self.fx:.1f}, fy={self.fy:.1f}, cx={self.cx:.1f}, cy={self.cy:.1f}"
            )

        except Exception as e:
            print(f"[BALL_DETECT] Failed to load camera calibration: {e}")
            print("[BALL_DETECT] Using default camera parameters")
            # Default camera matrix for 640x480 resolution
            self.camera_matrix = np.array(
                [[500, 0, 320], [0, 500, 240], [0, 0, 1]], dtype=float
            )
            self.dist_coeffs = np.zeros(5)
            self.fx = self.fy = 500
            self.cx, self.cy = 320, 240

    def _calculate_3d_position(self, center_px, radius_px):
        """Calculate 3D position of ball using camera intrinsics and known ball size.

        Args:
            center_px (tuple): Ball center in pixels (x, y)
            radius_px (float): Ball radius in pixels

        Returns:
            tuple: (x, y, z) position in meters relative to camera
        """
        if self.camera_matrix is None:
            return None

        # Calculate distance to ball using known ball diameter and observed radius
        # Distance = (real_diameter * focal_length) / (2 * observed_radius_pixels)
        focal_length = (self.fx + self.fy) / 2  # Average focal length
        z = (self.ball_diameter_m * focal_length) / (2 * radius_px)

        # Convert pixel coordinates to camera coordinates
        x_px, y_px = center_px
        x = (x_px - self.cx) * z / self.fx
        y = (y_px - self.cy) * z / self.fy

        return (x, y, z)

    def detect_ball(self, frame, apriltag_position=None):
        """Detect ball in frame and return detection results.

        Args:
            frame: Input BGR image frame
            apriltag_position (dict, optional): AprilTag pose data with keys:
                - 'x': x position in meters (camera frame)
                - 'y': y position in meters (camera frame)
                - 'z': z position in meters (camera frame)
                - 'center': (x, y) pixel coordinates of tag center

        Returns:
            found (bool): True if ball detected
            center (tuple): (x, y) pixel coordinates of ball center
            radius (float): Ball radius in pixels
            position_m (tuple): Ball 3D position (x, y, z) in meters from camera
            filtered_vel (tuple): Ball velocity (vx, vy, vz) in m/s from EKF
        """
        # Time update for EKF predict
        now = time.time()
        dt = now - self.last_time if hasattr(self, "last_time") else 0.0
        self.last_time = now
        # Always predict (even if no detection this frame)
        try:
            self.ekf.predict(dt)
        except Exception as e:
            # If EKF not initialized or other, ignore
            print(f"[BALL_DETECT] EKF predict error: {e}")
            pass

        # Convert frame from BGR to HSV color space for better color filtering
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create binary mask using HSV color bounds
        mask = cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)

        # Clean up mask using morphological operations
        mask = cv2.erode(mask, None, iterations=2)  # Remove noise
        mask = cv2.dilate(mask, None, iterations=2)  # Fill gaps

        # Find all contours in the cleaned mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # No detection this frame: return predicted position (if available)
            if self.ekf._initialized:
                pred_pos, _ = self.ekf.get_state()
                # Use predicted position and zero radius
                return False, None, None, pred_pos, (0.0, 0.0, 0.0)
            return False, None, None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

        # Select the largest contour (assumed to be the ball)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get minimum enclosing circle around the contour
        ((x, y), radius) = cv2.minEnclosingCircle(largest_contour)

        # Filter out detections that are too small or too large
        if radius < self.min_radius or radius > self.max_radius:
            # detection rejected; return predicted pos if available
            if self.ekf._initialized:
                pred_pos, _ = self.ekf.get_state()
                return False, None, None, pred_pos, (0.0, 0.0, 0.0)
            return False, None, None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

        # Calculate 3D position using camera intrinsics and known ball size
        position_3d = self._calculate_3d_position((x, y), radius)
        if position_3d is None:
            # Fallback to old 2D method if no camera calibration
            center_x = frame.shape[1] // 2
            center_y = frame.shape[0] // 2
            normalized_x = (x - center_x) / center_x
            normalized_y = (y - center_y) / center_y
            position_3d = (
                normalized_x * self.scale_factor,
                normalized_y * self.scale_factor,
                0.0,
            )

        # Update EKF with the new measurement (3D position)
        try:
            self.ekf.update(position_3d)
            # Use filtered position for outputs
            filtered_pos, filtered_vel = self.ekf.get_state()
        except Exception:
            filtered_pos = position_3d
            filtered_vel = (0.0, 0.0, 0.0)

        return (
            True,
            (int(x), int(y)),
            radius,
            filtered_pos,
            filtered_vel,
        )

    # Project a 3D point in camera coordinates to pixel coordinates
    def _project_3d_to_pixel(self, point3):
        """Project a 3D point.

        Project a 3D point (X,Y,Z) in camera frame to pixel coords (u,v).

        Args:
            point3 (tuple): 3D point (X, Y, Z) in meters
        Returns None if projection is invalid (e.g. Z<=0 or intrinsics missing).
        """
        if point3 is None:
            return None
        X, Y, Z = point3
        if Z <= 0 or self.camera_matrix is None:
            return None
        try:
            u = int(round((X * self.fx) / Z + self.cx))
            v = int(round((Y * self.fy) / Z + self.cy))
            return (u, v)
        except Exception:
            return None

    def draw_detection(self, frame, show_info=True, apriltag_position=None):
        """Detect ball and draw detection overlay on frame.

        Args:
            frame: Input BGR image frame
            show_info (bool): Whether to display position information text
            apriltag_position (dict, optional): AprilTag pose data for distance calculation

        Returns:
            frame_with_overlay: Frame with detection drawn
            found: True if ball detected
            position_m: Ball 3D position in meters
            distance_to_tag: Distance to AprilTag in meters (None if no tag)
        """
        # Perform ball detection with optional tag distance calculation
        (
            found,
            center,
            radius,
            position_m,
            distance_to_tag,
            filtered_vel,
        ) = self.detect_ball(frame, apriltag_position=apriltag_position)

        # Create overlay copy for drawing
        overlay = frame.copy()

        # Visualize detected / filtered ball
        if found:
            # Draw circle around detected ball
            cv2.circle(overlay, center, int(radius), (0, 255, 0), 2)  # Green circle
            cv2.circle(overlay, center, 3, (0, 255, 0), -1)  # Green center dot

            if show_info:
                # Display ball position information
                cv2.putText(
                    overlay,
                    f"3D: x={position_m[0]:.3f}m y={position_m[1]:.3f}m z={position_m[2]:.3f}m",
                    (center[0] - 60, center[1] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 255),
                    1,
                )

                # Display distance to AprilTag if available
                if distance_to_tag is not None:
                    cv2.putText(
                        overlay,
                        f"dist to tag id {apriltag_position['tag_id']}: {distance_to_tag:.3f}m",
                        (center[0] - 50, center[1] + 0),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 0),
                        1,
                    )

        # ---: predicted future-state visualization using EKF state ---
        if self.ekf._initialized:
            # get filtered position & velocity from EKF
            filtered_pos, filtered_vel = self.ekf.get_state()
            # project filtered current position to pixel (for reference)
            current_px = self._project_3d_to_pixel(filtered_pos)

            pred_pts = []
            for t in self.future_time_steps:
                # simple constant-velocity prediction
                fut = (
                    filtered_pos[0] + filtered_vel[0] * t,
                    filtered_pos[1] + filtered_vel[1] * t,
                    filtered_pos[2] + filtered_vel[2] * t,
                )
                px = self._project_3d_to_pixel(fut)
                pred_pts.append((t, fut, px))

            # Draw predicted points and connecting polyline
            prev_px = None
            for idx, (t, fut3, px) in enumerate(pred_pts):
                if px is None:
                    prev_px = None
                    continue
                # color gradient from blue -> red (earlier -> later)
                col = (
                    int(255 * idx / max(1, len(pred_pts) - 1)),
                    100,
                    int(255 * (1 - idx / max(1, len(pred_pts) - 1))),
                )
                # small circle for predicted point
                cv2.circle(overlay, px, 4, col, -1)
                # label time in ms
                cv2.putText(
                    overlay,
                    f"{int(t*1000)}ms",
                    (px[0] + 6, px[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    col,
                    1,
                )
                # connecting line
                if prev_px is not None:
                    cv2.line(overlay, prev_px, px, (200, 200, 200), 1)
                prev_px = px

            # draw line from current predicted pixel to first future point
            if current_px is not None and pred_pts:
                first_px = pred_pts[0][2]
                if first_px is not None:
                    cv2.line(overlay, current_px, first_px, (0, 200, 200), 1)
                    # also mark the filtered current pixel (cyan)
                    cv2.circle(overlay, current_px, 5, (0, 200, 200), 1)
        # --- end prediction visualization ---

        return overlay, found, position_m, distance_to_tag


# Legacy function for backward compatibility with existing code
def detect_ball_x(frame):
    """Legacy function that matches the old ball_detection.py interface.

    This function maintains compatibility with existing code that expects
    the original function signature and return format.

    Args:
        frame: Input BGR image frame

    Returns:
        found (bool): True if ball detected
        x_normalized (float): Normalized x position (-1 to +1)
        vis_frame (array): Frame with detection overlay
    """
    # Create detector instance using default config
    detector = BallDetector()

    # Get detection results with visual overlay
    vis_frame, found, position_m, _ = detector.draw_detection(frame)

    if found:
        # Convert back to normalized coordinates for legacy compatibility
        # position_m is a tuple (x,y,z)
        x_normalized = (
            position_m[0] / detector.scale_factor if detector.scale_factor != 0 else 0.0
        )
        x_normalized = np.clip(x_normalized, -1.0, 1.0)  # Ensure within bounds
    else:
        x_normalized = 0.0

    return found, x_normalized, vis_frame


# For testing/calibration when run directly
def main():
    """Test ball detection with current config."""
    detector = BallDetector()
    cap = cv2.VideoCapture(0)  # Use default camera

    print("Ball Detection Test")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Resize frame for consistent processing
        frame = cv2.resize(frame, (640, 480))

        # Get detection results with overlay
        vis_frame, found, position_m, distance_to_tag = detector.draw_detection(frame)

        # Show detection info in console
        if found:
            x, y, z = position_m
            print(f"Ball detected at 3D position: x={x:.4f}m, y={y:.4f}m, z={z:.4f}m")
            if distance_to_tag is not None:
                print(f"Distance to AprilTag: {distance_to_tag:.4f}m")

        # Display frame with detection overlay
        cv2.imshow("Ball Detection Test", vis_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
