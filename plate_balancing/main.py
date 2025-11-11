"""Main file for all the fun stuff."""
import queue
import threading
import time

import cv2
from aprilTag_plate_detector import AprilTagPlateDetector
from ball_distance import BallDistanceCalculator
from ball_plate_detection import BallDetector
from inverse_kinematics import InverseKinematics
from pid_controller import StewartPlatformController


class CameraThread(threading.Thread):
    """Threaded camera reader that continuously grabs frames."""

    def __init__(self, camera_index=1, queue_size=2):
        """Initialize the camera thread.

        Args:
            camera_index: Index of the camera to use.
            queue_size: Maximum number of frames to store in the queue.
        """
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera.")

        self.ball_detector = BallDetector()
        self.tag_detector = AprilTagPlateDetector()
        self.ball_distance_calc = BallDistanceCalculator(detector=self.tag_detector)
        # Thread-safe queues for data exchange
        self.ball_kinematics_queue = queue.Queue(maxsize=queue_size)
        self.apriltag_angles_queue = queue.Queue(maxsize=queue_size)
        self.running = True

    def run(self):
        """Continuously capture frames and perform detection."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue

            # --- AprilTag detection ---
            pitch, roll, used_previous, tag_positions = self.detect_apriltags(frame)
            if pitch is not None and roll is not None:
                self._put_in_queue(
                    self.apriltag_angles_queue,
                    {"pitch": pitch, "roll": roll, "used_previous": used_previous},
                )
            # --- Ball detection ---
            ball_data = self.detect_ball(frame, apriltag_position=tag_positions)
            if ball_data is not None:
                self._put_in_queue(self.ball_kinematics_queue, ball_data)

    def detect_ball(self, frame, apriltag_position=None):
        """Detect the ball in the frame.

        Args:
            frame: The camera frame.
            apriltag_position: The position of the AprilTag, if detected.

        Returns:
            Ball position data or None if not detected.
        """
        # Perform ball detection with optional tag distance calculation
        (
            found,
            center,
            radius,
            position_m,
            filtered_vel,
        ) = self.ball_detector.detect_ball(frame, apriltag_position=apriltag_position)
        ball_data = {
            "ball_found": found,
            "center": center,
            "radius": radius,
            "position_m": position_m,
            "filtered_vel": filtered_vel,
        }
        if found and apriltag_position is not None:
            results = self.ball_distance_calc.process_frame(
                frame, apriltag_position, ball_data=ball_data
            )
            if results["error_message"]:
                print(f"Ball distance error: {results['error_message']}")
            x_dist = results["xy_analysis"]["x_distance"]
            y_dist = results["xy_analysis"]["y_distance"]
            x_vel = results["xy_analysis"]["x_velocity"]
            y_vel = results["xy_analysis"]["y_velocity"]
            return {
                "x_dist": x_dist,
                "y_dist": y_dist,
                "x_vel": x_vel,
                "y_vel": y_vel,
            }
        return None

    def detect_apriltags(self, frame):
        """Detect AprilTags in the frame.

        Args:
            frame: The camera frame.

        Returns:
            AprilTag pose data or None if not detected.
        """
        pitch, roll, used_previous, tag_positions = self.tag_detector.process_frame(
            frame
        )
        if pitch is not None and roll is not None:
            return pitch, roll, used_previous, tag_positions

        return None, None, False, None

    def _put_in_queue(self, queue_obj, data):
        """Put data in queue, dropping old data if queue is full."""
        try:
            if queue_obj.full():
                queue_obj.get_nowait()  # Remove old data
            queue_obj.put_nowait(data)
        except queue.Empty:
            pass

    def get_ball_kinematics(self):
        """Return the newest ball data if available, else None."""
        try:
            return self.ball_kinematics_queue.get_nowait()
        except queue.Empty:
            return None

    def get_apriltag_angles(self):
        """Return the newest apriltag data if available, else None."""
        try:
            return self.apriltag_angles_queue.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        """Stop the camera thread and release resources."""
        self.running = False
        self.cap.release()


class ControllerThread(threading.Thread):
    """Threaded controller for plate balancing."""

    def __init__(self, camera_thread, ik_solver, dt=0.05):
        """Initialize the controller thread.

        Args:
            camera_thread: Reference to camera thread for data access.
            ik_solver: Instance of InverseKinematics.
            dt: Time between control updates in seconds.
        """
        super().__init__(daemon=True)
        self.controller = StewartPlatformController()
        self.camera_thread = camera_thread
        self.ik = ik_solver
        self.dt = dt
        self.running = True

    def run(self):
        """Continuously update the controller."""
        while self.running:
            # Get latest data from camera thread
            ball_data = self.camera_thread.get_ball_kinematics()
            apriltag_data = self.camera_thread.get_apriltag_angles()

            if ball_data is not None and apriltag_data is not None:
                # Extract data
                x_dist = ball_data["x_dist"]
                y_dist = ball_data["y_dist"]
                x_vel = ball_data["x_vel"]
                y_vel = ball_data["y_vel"]
                pitch = apriltag_data["pitch"]
                roll = apriltag_data["roll"]
                # Update controller
                pitch, roll, z = self.controller.update_control(
                    ball_kinematics=(x_dist, y_dist, x_vel, y_vel),
                    plate_pitch=pitch,
                    plate_roll=roll,
                    dt=self.dt,
                )

                # Calculate and move servo angles
                self.ik.move_to_position(pitch, roll, z)

            time.sleep(self.dt)

    def stop(self):
        """Stop the controller thread."""
        self.running = False


def main():
    """Main function to run the camera and detectors."""
    inverse_kinematics = InverseKinematics()
    # Start threads
    cam_thread = CameraThread()
    controller_thread = ControllerThread(cam_thread, inverse_kinematics)

    cam_thread.start()
    controller_thread.start()
    print("Threads started. Press 'q' to quit.")

    try:
        while True:
            # Main thread can still access data for visualization
            ball_data = cam_thread.get_ball_kinematics()
            apriltag_data = cam_thread.get_apriltag_angles()

            if ball_data:
                print(f"Ball data available: {ball_data}")
            if apriltag_data:
                print(f"AprilTag data available: {apriltag_data}")

            time.sleep(0.1)
            # Add your visualization logic here

    except KeyboardInterrupt:
        print("[INFO] Shutting down...")
    finally:
        controller_thread.stop()
        cam_thread.stop()
        controller_thread.join(timeout=1)
        cam_thread.join(timeout=1)
        print("[INFO] Threads stopped cleanly.")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
