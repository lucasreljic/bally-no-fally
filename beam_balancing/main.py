"""Ball detection module for beam balancing system.

This module provides functionality to detect and track a yellow ball using
computer vision techniques with OpenCV.
"""

import cv2 as cv
from apriltags_detector import AprilTagDetector
from ball_detection import BallDetector

# '***' stands for things to modify for your own webcam, display, and ball if needed


def detect_yellow_ball():
    """Detect and track a yellow ball using webcam input.

    This function captures video from the default webcam, processes each frame
    to detect yellow objects, and displays the video feed with detected ball
    highlighted. The ball's position is printed to the console.

    The function runs in a loop until 'q' is pressed to quit.
    """
    # Start capturing video from the webcam. If multiple webcams connected, you may use 1,2, etc.
    cap = cv.VideoCapture(0)
    # *1 CAP_PROP_FPS sets the frame rate of the webcam to 30 fps here
    cap.set(cv.CAP_PROP_FPS, 30)
    detector = AprilTagDetector("camera_calibration.npz", tag_size_m=0.037)
    ball_detector = BallDetector()

    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        alpha = (
            0.9  # Contrast control (1.0 means no change, >1 increases, <1 decreases)
        )
        beta = 10  # Brightness control (positive increases, negative decreases)

        # Apply the contrast and brightness adjustment
        frame = cv.convertScaleAbs(frame, alpha=alpha, beta=beta)
        # Convert the frame from BGR to HSV color space to easily identify a colour
        # Get detection results with overlay
        pose_data = detector.detect_apriltag_poses(frame)
        tag_position = pose_data[0] if pose_data else None

        vis_frame, found, position_m, distance_to_tag = ball_detector.draw_detection(
            frame, apriltag_position=tag_position
        )
        frame_with_overlay = detector.draw_detection_overlay(vis_frame, pose_data)
        if not found or distance_to_tag is None:
            print("Ball not found")
        # for pose in pose_data:
        #     print(f"Tag {pose['tag_id']}: x={pose['x']:.3f}m, y={pose['y']:.3f}m, z={pose['z']:.3f}m")
        # Display the resulting frame with resize
        scale_percent = 50
        width = int(frame_with_overlay.shape[1] * scale_percent / 100)
        height = int(frame_with_overlay.shape[0] * scale_percent / 100)
        frame_with_overlay = cv.resize(frame_with_overlay, (width, height))
        cv.imshow("frame", frame_with_overlay)

        # Break the loop when 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord("q"):
            break

    # Release the capture when everything is done
    cap.release()
    # Close all windows
    cv.destroyAllWindows()


if __name__ == "__main__":
    detect_yellow_ball()
