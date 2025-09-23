"""PID control script for beam balancing via HSV tracking and stepper control.

This module:
- Captures frames from a camera.
- Detects the ball, tip, and center markers in HSV color space.
- Computes the angular error between the ball and tip vectors about the center.
- Runs a PID loop (with trackbars to tune Kp/Ki/Kd live) to rotate a stepper motor
  until the error is within a deadband.
"""

from time import sleep
import math

import cv2 as cv
import numpy as np
from gpiozero import DigitalOutputDevice

# --- Motor setup ---
ENA = DigitalOutputDevice(17)  # 3.3V power pin (driver enable, active-low)
DIR = DigitalOutputDevice(27)  # GPIO 27 - Motor direction
PUL = DigitalOutputDevice(22)  # GPIO 22 - Motor step pulse
ENA.off()  # Enable driver (active low)


def step(delay: float = 0.005) -> None:
    """Emit a single step pulse to the stepper driver.

    Args:
        delay: Half-period delay (seconds) to set the pulse width / step rate.
    """
    PUL.on()
    sleep(delay)
    PUL.off()
    sleep(delay)


def rotate_steps(n_steps: int, direction: int, delay: float = 0.005) -> None:
    """Rotate the motor a given number of steps in a given direction.

    Args:
        n_steps: Number of step pulses to send (>= 0).
        direction: 1 for clockwise, 0 for counter-clockwise (driver-dependent).
        delay: Per-step half-period delay in seconds (see `step`).
    """
    DIR.value = direction
    for _ in range(n_steps):
        step(delay)


# --- HSV color ranges ---
ball_lower = np.array([8, 140, 88])
ball_upper = np.array([50, 255, 255])

center_lower = np.array([75, 100, 100])
center_upper = np.array([150, 255, 255])

tip_lower = np.array([55, 60, 19])
tip_upper = np.array([77, 255, 255])

kernel = np.ones((5, 5), np.uint8)

deg_per_step = 0.9
deadband = 5.0
step_delay = 0.005

# --- Create control window with sliders ---
cv.namedWindow("PID Control")


def nothing(_: int) -> None:
    """No-op callback for OpenCV trackbars (required signature)."""
    return


cv.createTrackbar("Kp x100", "PID Control", 24, 30, nothing)
cv.createTrackbar("Ki x1000", "PID Control", 0, 50, nothing)
cv.createTrackbar("Kd x100", "PID Control", 4, 50, nothing)

# --- Camera setup ---
cap = cv.VideoCapture(0, cv.CAP_V4L2)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 480)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv.CAP_PROP_FPS, 30)


def find_largest_blob_center(mask: np.ndarray) -> np.ndarray | None:
    """Find the centroid of the largest contour in a binary mask.

    Small blobs are rejected to reduce noise.

    Args:
        mask: Binary mask (uint8) where 255 marks the regions of interest.

    Returns:
        2D numpy array [cx, cy] for the blob center, or None if not found.
    """
    cnts, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if cnts:
        largest = max(cnts, key=cv.contourArea)
        if cv.contourArea(largest) > 100:
            M = cv.moments(largest)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                return np.array([cx, cy])
    return None


# PID state
integral = 0.0
prev_error = 0.0

print("Starting main loop. Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv.resize(frame, (480, 480))
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    mask_ball = cv.inRange(hsv, ball_lower, ball_upper)
    mask_tip = cv.inRange(hsv, tip_lower, tip_upper)
    mask_center = cv.inRange(hsv, center_lower, center_upper)

    mask_ball = cv.morphologyEx(mask_ball, cv.MORPH_OPEN, kernel)
    mask_tip = cv.morphologyEx(mask_tip, cv.MORPH_OPEN, kernel)
    mask_center = cv.morphologyEx(mask_center, cv.MORPH_OPEN, kernel)

    ball_pos = find_largest_blob_center(mask_ball)
    tip_pos = find_largest_blob_center(mask_tip)
    center_pos = find_largest_blob_center(mask_center)

    if ball_pos is not None:
        cv.circle(frame, tuple(ball_pos), 6, (0, 255, 255), -1)
    if tip_pos is not None:
        cv.circle(frame, tuple(tip_pos), 6, (0, 255, 0), -1)
    if center_pos is not None:
        cv.circle(frame, tuple(center_pos), 6, (255, 0, 255), -1)

    if ball_pos is not None and tip_pos is not None and center_pos is not None:
        v_ball = ball_pos - center_pos
        v_tip = tip_pos - center_pos

        angle_ball = math.atan2(v_ball[1], v_ball[0])
        angle_tip = math.atan2(v_tip[1], v_tip[0])

        # Wrap difference to (-pi, pi]
        angle_error_rad = (angle_ball - angle_tip + math.pi) % (2 * math.pi) - math.pi
        angle_error_deg = math.degrees(angle_error_rad)

        # --- PID Calculation (live-tuned via trackbars) ---
        Kp = cv.getTrackbarPos("Kp x100", "PID Control") / 100.0
        Ki = cv.getTrackbarPos("Ki x1000", "PID Control") / 1000.0
        Kd = cv.getTrackbarPos("Kd x100", "PID Control") / 100.0

        integral += angle_error_deg
        derivative = angle_error_deg - prev_error
        prev_error = angle_error_deg

        pid_output = Kp * angle_error_deg + Ki * integral + Kd * derivative
        steps = int(abs(pid_output) / deg_per_step)

        if abs(angle_error_deg) > deadband and steps > 0:
            direction = 1 if pid_output > 0 else 0
            print(
                f"PID → Error: {angle_error_deg:.2f}, Steps: {steps}, "
                f"Dir: {'CW' if direction else 'CCW'}"
            )
            rotate_steps(steps, direction, step_delay)
        else:
            print("Within deadband or too small; no movement.")

        cv.putText(
            frame,
            f"Error: {angle_error_deg:.1f} deg",
            (10, 30),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    cv.imshow("Frame", frame)
    cv.imshow("Mask Ball", mask_ball)
    cv.imshow("Mask Tip", mask_tip)
    cv.imshow("Mask Center", mask_center)

    if cv.waitKey(1) & 0xFF == ord("q"):
        break

    sleep(0.01)

cap.release()
ENA.on()
cv.destroyAllWindows()
