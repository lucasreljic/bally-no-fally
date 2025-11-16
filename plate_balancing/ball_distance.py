"""Ball Distance Calculation Module.

This module integrates AprilTag detection and ball detection to calculate:
1. Distance between ball and plate center
2. Distance between ball and tag ID 5
3. Angle formed by tag 5-center line and ball-center line
4. X,Y distances from ball to center using trigonometry
"""

import math

import cv2
import numpy as np
from aprilTag_plate_detector import AprilTagPlateDetector
from ball_plate_detection import BallDetector


class BallDistanceCalculator:
    """Calculate distances and angles between ball, plate center, and tag 5."""

    def __init__(
        self,
        camera_calibration_path="camera_calibration.npz",
        ball_config_path="ball_config.json",
        tag_size=0.032,
        detector=None,
        ball_det=None,
    ):
        """Initialize the distance calculator.

        Args:
            camera_calibration_path (str): Path to camera calibration file
            ball_config_path (str): Path to ball detection config file
            tag_size (float): Tag side length in meters
            detector (AprilTagPlateDetector): Optional pre-initialized tag detector
            ball_det (BallDetector): Optional pre-initialized ball detector
        """
        # Initialize AprilTag detector
        self.tag_detector = (
            detector
            if detector
            else AprilTagPlateDetector(
                camera_calibration_path=camera_calibration_path,
                tag_size=tag_size,
                avg_frames=1,  # Use 1 for real-time calculations
            )
        )

        # Initialize ball detector
        self.ball_detector = (
            ball_det
            if ball_det
            else BallDetector(
                config_file=ball_config_path,
                calibration_file=camera_calibration_path,
            )
        )

        # Load camera parameters for 3D-to-2D projection
        self.camera_params = self.tag_detector.camera_params
        self.fx, self.fy, self.cx, self.cy = self.camera_params

    def project_3d_to_2d(self, point_3d):
        """Project 3D point to 2D pixel coordinates.

        Args:
            point_3d (tuple): 3D point (x, y, z) in meters

        Returns:
            tuple: 2D pixel coordinates (u, v) or None if invalid
        """
        if point_3d is None or len(point_3d) != 3:
            return None

        x, y, z = point_3d
        if z <= 0:  # Point behind camera
            return None

        # Perspective projection
        u = self.fx * (x / z) + self.cx
        v = self.fy * (y / z) + self.cy

        return (int(u), int(v))

    def calculate_distance_2d(self, point1, point2):
        """Calculate 2D Euclidean distance between two points (X,Y only).

        Args:
            point1 (tuple): First 3D point (x, y, z) - only x,y used
            point2 (tuple): Second 3D point (x, y, z) - only x,y used

        Returns:
            float: 2D distance in meters (X,Y plane only)
        """
        if point1 is None or point2 is None:
            return None

        x1, y1 = point1[0], point1[1]
        x2, y2 = point2[0], point2[1]

        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def calculate_angle_between_vectors_2d(self, center, tag5_pos, ball_pos):
        """Calculate angle between tag5-center vector and ball-center vector (2D only).

        Args:
            center (tuple): Center position (x, y, z) - only x,y used
            tag5_pos (tuple): Tag 5 position (x, y, z) - only x,y used
            ball_pos (tuple): Ball position (x, y, z) - only x,y used

        Returns:
            float: Angle in degrees between the two vectors in X,Y plane
        """
        if center is None or tag5_pos is None or ball_pos is None:
            return None

        # Create 2D vectors from center to tag5 and center to ball (X,Y only)
        center_2d = np.array([center[0], center[1]])
        tag5_2d = np.array([tag5_pos[0], tag5_pos[1]])
        ball_2d = np.array([ball_pos[0], ball_pos[1]])

        center_to_tag5 = tag5_2d - center_2d
        center_to_ball = ball_2d - center_2d

        # Calculate angle using dot product
        dot_product = np.dot(center_to_tag5, center_to_ball)
        mag_tag5 = np.linalg.norm(center_to_tag5)
        mag_ball = np.linalg.norm(center_to_ball)

        if mag_tag5 == 0 or mag_ball == 0:
            return None

        # Calculate angle in radians, then convert to degrees
        cos_angle = dot_product / (mag_tag5 * mag_ball)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # Ensure within valid range
        angle_rad = math.acos(cos_angle)
        angle_deg = math.degrees(angle_rad)

        return angle_deg

    def calculate_xy_distances_from_center(
        self, center, ball_pos, tag5_pos, ball_velocity=None
    ):
        """Calculate X,Y distances from ball to center using trigonometry (2D only).

        Uses the plate coordinate system where:
        - X-axis is aligned with the direction from center to tag 5 (X,Y plane only)
        - Y-axis is perpendicular to X-axis in the X,Y plane

        Args:
            center (tuple): Center position (x, y, z) - only x,y used
            ball_pos (tuple): Ball position (x, y, z) - only x,y used
            tag5_pos (tuple): Tag 5 position (x, y, z) - only x,y used
            ball_velocity (tuple): Ball velocity (vx, vy, vz) - only vx,vy used

        Returns:
            dict: Contains x_distance, y_distance, total_distance, angle, velocity components (all 2D)
        """
        if center is None or ball_pos is None or tag5_pos is None:
            return None

        # Extract only X,Y components
        center_2d = np.array([center[0], center[1]])
        ball_2d = np.array([ball_pos[0], ball_pos[1]])
        tag5_2d = np.array([tag5_pos[0], tag5_pos[1]])

        # Create 2D coordinate system
        # X-axis: direction from center to tag 5 (X,Y plane)
        x_axis_2d = tag5_2d - center_2d
        x_axis_norm = np.linalg.norm(x_axis_2d)
        if x_axis_norm == 0:
            return None
        x_axis_2d = x_axis_2d / x_axis_norm

        # Y-axis: perpendicular to X-axis in the X,Y plane
        y_axis_2d = np.array([-x_axis_2d[1], x_axis_2d[0]])

        # Vector from center to ball (2D)
        center_to_ball_2d = ball_2d - center_2d

        # Calculate distances along plate axes
        x_distance = np.dot(center_to_ball_2d - 0.00, x_axis_2d)
        y_distance = np.dot(center_to_ball_2d - 0.00, y_axis_2d)
        total_distance_2d = np.linalg.norm(center_to_ball_2d)

        # Calculate angle between center-tag5 direction and center-ball direction
        angle = self.calculate_angle_between_vectors_2d(center, tag5_pos, ball_pos)

        # Calculate velocity components in plate coordinate system
        x_velocity = y_velocity = total_velocity_2d = None
        if ball_velocity is not None:
            ball_velocity_2d = np.array([ball_velocity[0], ball_velocity[1]])
            x_velocity = np.dot(ball_velocity_2d, x_axis_2d)
            y_velocity = np.dot(ball_velocity_2d, y_axis_2d)
            total_velocity_2d = np.linalg.norm(ball_velocity_2d)

        return {
            "x_distance": x_distance,  # Distance along tag5 direction
            "y_distance": y_distance,  # Distance perpendicular to tag5 direction
            "total_distance_2d": total_distance_2d,  # 2D distance on plate
            "angle_degrees": angle,
            "x_velocity": x_velocity,  # Velocity along tag5 direction
            "y_velocity": y_velocity,  # Velocity perpendicular to tag5 direction
            "total_velocity_2d": total_velocity_2d,  # 2D velocity magnitude
        }

    def process_frame(self, frame, tag_positions, plate_center_3d, ball_data=None):
        """Process frame to detect ball and calculate all distances/angles.

        Args:
            frame: Input BGR image frame
            tag_positions: Dictionary of detected tag positions from AprilTag detector
            ball_data: Dictionary of detected ball data (if available)

        Returns:
            dict: Complete analysis results including positions, distances, and angles
        """
        # Get detected tag IDs from tag_positions
        detected_tag_ids = set(tag_positions.keys())
        plate_center_2d = self.tag_detector.find_2D_plate_center(plate_center_3d)

        # Get ball detection results
        ball_found = ball_data["ball_found"] if ball_data else False
        ball_center_2d = ball_data["center"] if ball_data else None
        ball_radius = ball_data["radius"] if ball_data else None
        ball_pos_3d = ball_data["position_m"] if ball_data else None
        ball_velocity = ball_data["filtered_vel"] if ball_data else None

        # Initialize results dictionary
        results = {
            "tag_positions": tag_positions,
            "detected_tag_ids": list(detected_tag_ids),
            "plate_center_2d": plate_center_2d,
            "ball_found": ball_found,
            "ball_center_2d": ball_center_2d,
            "ball_radius": ball_radius,
            "ball_pos_3d": ball_pos_3d,
            "ball_velocity": ball_velocity,
            "distances": None,
            "error_message": None,
        }

        # Check if we have all required components
        if not ball_found:
            results["error_message"] = "Ball not detected"
            return results

        if 5 not in tag_positions:
            if 5 in self.tag_detector.last_known_positions:
                tag_positions[5] = self.tag_detector.last_known_positions[5]
            else:
                results["error_message"] = "Tag 5 not detected and no stored position"
                return results

        if plate_center_3d is None:
            results[
                "error_message"
            ] = "Insufficient platform tags for center calculation"
            return results

        # print(
        #     f"tags;  x: {plate_center_3d[0]:.3f}m, "
        #     f"y: {plate_center_3d[1]:.3f}m, "
        #     f"z: {plate_center_3d[2]:.3f}m, "
        # )
        tag5_pos_3d = tag_positions[5]

        # Calculate all distances and angles (2D only)
        ball_to_center_distance = self.calculate_distance_2d(
            ball_pos_3d, plate_center_3d
        )
        ball_to_tag5_distance = self.calculate_distance_2d(ball_pos_3d, tag5_pos_3d)
        xy_distances = self.calculate_xy_distances_from_center(
            plate_center_3d, ball_pos_3d, tag5_pos_3d, ball_velocity
        )

        # Store distance results
        results = {
            "ball_found": ball_found,
            "ball_center_2d": ball_center_2d,
            "ball_radius": ball_radius,
            "ball_velocity": ball_velocity,
            "ball_to_center_2d": ball_to_center_distance,
            "ball_to_tag5_2d": ball_to_tag5_distance,
            "plate_center_xy": (plate_center_3d[0], plate_center_3d[1]),
            "tag5_pos_xy": (tag5_pos_3d[0], tag5_pos_3d[1]),
            "ball_pos_xy": (ball_pos_3d[0], ball_pos_3d[1]),
            # Keep 3D positions for internal use (projection, etc.)
            "plate_center_3d": tuple(plate_center_3d),
            "tag5_pos_3d": tuple(tag5_pos_3d),
            "ball_pos_3d": tuple(ball_pos_3d),
            "plate_center_2d": plate_center_2d,
            "xy_analysis": xy_distances,
        }

        return results

    def draw_visualization(self, frame, results):
        """Draw visualization of distances and angles on the frame.

        Args:
            frame: Input BGR image frame
            results: Results dictionary from process_frame

        Returns:
            frame: Frame with visualization overlay
        """
        if results is None:
            if results["error_message"]:
                cv2.putText(
                    frame,
                    f"Error: {results['error_message']}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
            return frame

        distances = results
        xy_analysis = distances["xy_analysis"]

        # Draw ball detection with enhanced info
        if results["ball_found"] and results["ball_center_2d"]:
            ball_center = results["ball_center_2d"]
            ball_radius = results["ball_radius"]

            # Draw ball
            cv2.circle(frame, ball_center, int(ball_radius), (0, 255, 0), 2)
            cv2.circle(frame, ball_center, 3, (0, 255, 0), -1)

            # Draw lines from ball to center and tag 5 (if visible)
            if results["plate_center_2d"]:
                plate_center_2d = results["plate_center_2d"]
                cv2.line(frame, ball_center, plate_center_2d, (255, 0, 255), 2)

            # Project tag 5 to 2D and draw line
            tag5_2d = self.project_3d_to_2d(distances["tag5_pos_3d"])
            if tag5_2d:
                cv2.line(frame, ball_center, tag5_2d, (0, 255, 255), 2)

            # Display distance information (2D only)
            y_offset = 30
            info_texts = [
                f"Ball to Center (2D): {distances['ball_to_center_2d']:.3f}m",
                f"Ball to Tag5 (2D): {distances['ball_to_tag5_2d']:.3f}m",
                f"Ball (3D): {distances['ball_pos_3d'][2]:.3f}m",
                f"Tag5 (3D): {distances['tag5_pos_3d'][2]:.3f}m",
                f"X Distance: {xy_analysis['x_distance']:.3f}m",
                f"Y Distance: {xy_analysis['y_distance']:.3f}m",
                f"Angle: {xy_analysis['angle_degrees']:.1f}°",
            ]

            for i, text in enumerate(info_texts):
                cv2.putText(
                    frame,
                    text,
                    (10, y_offset + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    1,
                )

            # Draw coordinate system on plate
            if results["plate_center_2d"] and tag5_2d:
                center_2d = results["plate_center_2d"]

                # Draw X-axis (toward tag 5)
                cv2.arrowedLine(
                    frame, center_2d, tag5_2d, (0, 0, 255), 2, tipLength=0.3
                )
                cv2.putText(
                    frame, "X", tag5_2d, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
                )

                # Draw Y-axis (perpendicular to X)
                direction_x = np.array(tag5_2d) - np.array(center_2d)
                direction_x = direction_x / np.linalg.norm(direction_x) * 50
                direction_y = np.array([-direction_x[1], direction_x[0]])
                y_end = tuple((np.array(center_2d) + direction_y).astype(int))

                cv2.arrowedLine(frame, center_2d, y_end, (255, 0, 0), 2, tipLength=0.3)
                cv2.putText(
                    frame, "Y", y_end, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
                )
        return frame

    def run(self, camera_index=0):
        """Run the main detection and distance calculation loop.

        Args:
            camera_index (int): Camera device index
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Could not open camera.")
            return

        print("Ball Distance Calculator")
        print("Press 'q' to quit")
        print()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Camera frame not received.")
                    break

                # Process frame and calculate distances
                results = self.process_frame(frame)

                # Draw visualization
                vis_frame = self.draw_visualization(frame, results)

                # Display results in console (2D only)
                if results is not None:
                    distances = results
                    xy_analysis = distances["xy_analysis"]

                    print(
                        f"Ball-Center (2D): {distances['ball_to_center_2d']:.3f}m, "
                        f"Ball-Tag5 (2D): {distances['ball_to_tag5_2d']:.3f}m, "
                        f"X: {xy_analysis['x_distance']:.3f}m, "
                        f"Y: {xy_analysis['y_distance']:.3f}m, "
                        f"Angle: {xy_analysis['angle_degrees']:.1f}°"
                    )

                # Show frame
                cv2.imshow("Ball Distance Calculator", vis_frame)

                # Exit on 'q' key press
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()


def main():
    """Main function to run the ball distance calculator."""
    calculator = BallDistanceCalculator()
    calculator.run()


if __name__ == "__main__":
    main()
