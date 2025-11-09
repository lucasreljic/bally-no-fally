"""Main file for all the fun stuff."""
import queue
import threading
import time

import cv2
from ball_plate_detection import BallDetector


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
        self.frames = queue.Queue(maxsize=queue_size)
        self.running = True

    def run(self):
        """Continuously capture frames from the camera."""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                continue
            # Drop old frame if queue full
            if not self.frames.empty():
                try:
                    self.frames.get_nowait()
                except queue.Empty:
                    pass
            self.frames.put(frame)

    def get_latest(self):
        """Return the newest frame if available, else None."""
        if self.frames.empty():
            return None
        try:
            return self.frames.get_nowait()
        except queue.Empty:
            return None

    def stop(self):
        """Stop the camera thread and release resources."""
        self.running = False
        self.cap.release()


def main():
    """Main function to run the camera and detectors."""
    # Initialize detectors
    # plate_detector = AprilTagPlateDetector(camera_calibration_path="camera_calibration.npz")
    ball_detector = BallDetector(
        config_file="ball_config.json", calibration_file="camera_calibration.npz"
    )

    # Start camera thread
    cam_thread = CameraThread()
    cam_thread.start()
    print("Camera thread started. Press 'q' to quit.")

    try:
        while True:
            frame = cam_thread.get_latest()
            if frame is None:
                time.sleep(0.01)
                continue

            # --- AprilTag plate orientation ---
            # pitch, roll, using_stored = plate_detector.detect_from_frame(frame)
            # if pitch is not None and roll is not None:
            #     print(f"Plate: Pitch={pitch:.2f}°, Roll={roll:.2f}°{' (stored)' if using_stored else ''}")

            # --- Ball detection ---
            vis_frame, found, ball_pos, _ = ball_detector.draw_detection(frame)
            if found:
                x, y, z = ball_pos
                print(f"Ball: x={x:.3f}m y={y:.3f}m z={z:.3f}m")

            # Combine visualizations
            cv2.imshow("Stewart Platform Vision", vis_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cam_thread.stop()
        cv2.destroyAllWindows()
        print("Shutting down cleanly.")


if __name__ == "__main__":
    main()
