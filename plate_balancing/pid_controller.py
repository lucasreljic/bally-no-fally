"""PID control script for beam balancing via HSV tracking and stepper control.

This module:
- Captures frames from a camera.
- Detects the ball, tip, and center markers in HSV color space.
- Computes the angular error between the ball and tip vectors about the center.
- Runs a PID loop (with trackbars to tune Kp/Ki/Kd live) to rotate a stepper motor
  until the error is within a deadband.
"""

import argparse
import queue
import time
from threading import Thread

import cv2 as cv
import numpy as np
import serial
from ball_plate_detection import BallDetector


class PIDController:
    """PID controller for beam balancing using servo control."""

    def __init__(self, args, scale_factor=0.25):
        """Initialize controller and PID parameters."""
        
        # PID gains
        self.Kp_pitch = 0.08
        self.Ki_pitch = 0.04
        self.Kd_pitch = 0.015

        self.Kp_roll = 0.08
        self.Ki_roll = 0.04
        self.Kd_roll = 0.015

        self.Kp_z = 0.1
        self.Ki_z = 0.05
        self.Kd_z = 0.02

        self.visualization = not args.no_vis
        self.scale_factor = scale_factor
        self.scale_error = 100

        # Deadbands for PID
        self.deadband_pitch = 0.005
        self.deadband_roll = 0.005
        self.deadband_z = 0.01

        # Internal PID state
        self.integral_pitch = 0
        self.prev_error_pitch = 0

        self.integral_roll = 0
        self.prev_error_roll = 0

        self.integral_z = 0
        self.prev_error_z = 0

        # Thread-safe queue for ball position (from camera)
        self.position_queue = queue.Queue(maxsize=1)
        self.running = False
    
    def update_pid(self, position, dt=0.033):
        """Perform PID calculation and return control output."""
        error = self.setpoint - position  # Compute error
        if abs(error) < self.deadband:
            error = 0
        error = error * self.scale_error  # Scale error for easier tuning (if needed)

        # Proportional term
        P_val = self.Kp * error
        # Integral term accumulation
        if self.I_range < abs(error):
            self.integral = 0
        else:
            self.integral += error * dt
        I_val = self.Ki * self.integral
        # Derivative term calculation
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        D_val = self.Kd * derivative
        self.prev_error = error
        # PID output (limit to safe beam range)
        output = P_val + I_val + D_val
        output = np.clip(output, -15, 15)

        return output

    def camera_thread(self):
        """Dedicated thread for video capture and ball detection."""
        print("start camera thread")
        cap = cv.VideoCapture(0)
        cap.set(cv.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv.CAP_PROP_FPS, 30)
        detector = AprilTagDetector("camera_calibration.npz", tag_size_m=0.014)
        ball_detector = BallDetector()
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                continue
            # Detect ball position in frame
            alpha = 1.1  # Contrast control (1.0 means no change, >1 increases, <1 decreases)
            beta = -25  # Brightness control (positive increases, negative decreases)

            # Apply the contrast and brightness adjustment
            frame = cv.convertScaleAbs(frame, alpha=alpha, beta=beta)
            # Convert the frame from BGR to HSV color space to easily identify a colour
            # Get detection results with overlay
            pose_data = detector.detect_apriltag_poses(frame)
            tag_position = pose_data[0] if pose_data else None

            vis_frame, found, _, distance_to_tag = ball_detector.draw_detection(
                frame, apriltag_position=tag_position
            )

            if found and distance_to_tag:
                # Convert normalized to meters using scale
                position_m = (distance_to_tag - 0.038) - self.length_beam / 2
                # Always keep latest measurement only
                try:
                    if self.position_queue.full():
                        self.position_queue.get_nowait()
                    self.position_queue.put_nowait(position_m)
                except Exception:
                    print("[CAMERA] Could not add position to queue")
            elif len(pose_data) < 1:
                print("[CAMERA] Could not find Apriltag")
            else:
                print("[CAMERA] Could not find ball")
            # Show processed video with overlays (comment out for speed)
            # Live preview for debugging
            if self.visualization:
                frame_with_overlay = detector.draw_detection_overlay(
                    vis_frame, pose_data
                )
                scale_percent = 50
                width = int(frame_with_overlay.shape[1] * scale_percent / 100)
                height = int(frame_with_overlay.shape[0] * scale_percent / 100)
                frame_with_overlay = cv.resize(frame_with_overlay, (width, height))
                cv.imshow("frame", frame_with_overlay)

            if cv.waitKey(1) & 0xFF == 27:  # ESC exits
                self.running = False
                break
        cap.release()
        cv.destroyAllWindows()

    def control_thread(self):
        """Compute PID for pitch, roll, Z and send to inverse kinematics."""

        setpoint_pitch = 0.0
        setpoint_roll = 0.0
        setpoint_z = 0.12  # Example desired height (meters)

        prev_time = time.time()

        while self.running:
            try:
                # Wait for latest position
                position = self.position_queue.get(timeout=0.03)
                current_time = time.time()
                dt = current_time - prev_time
                prev_time = current_time

                # Example: map position to pitch/roll/Z measurements
                # Here, you can implement your own mapping from ball displacement to platform axes
                measured_pitch = position * 0.5  # placeholder
                measured_roll = position * -0.3  # placeholder
                measured_z = 0.12                 # placeholder

                # PID calculations
                pitch_out, self.integral_pitch, self.prev_error_pitch = self.update_pid(
                    setpoint_pitch, measured_pitch,
                    self.Kp_pitch, self.Ki_pitch, self.Kd_pitch,
                    self.integral_pitch, self.prev_error_pitch,
                    self.deadband_pitch, dt
                )

                roll_out, self.integral_roll, self.prev_error_roll = self.update_pid(
                    setpoint_roll, measured_roll,
                    self.Kp_roll, self.Ki_roll, self.Kd_roll,
                    self.integral_roll, self.prev_error_roll,
                    self.deadband_roll, dt
                )

                z_out, self.integral_z, self.prev_error_z = self.update_pid(
                    setpoint_z, measured_z,
                    self.Kp_z, self.Ki_z, self.Kd_z,
                    self.integral_z, self.prev_error_z,
                    self.deadband_z, dt
                )

                # Send PID outputs to Stewart platform controller
                set_platform_angles(pitch_out, roll_out, z_out)

                print(f"[CONTROL] Pitch: {pitch_out:.3f}, Roll: {roll_out:.3f}, Z: {z_out:.3f}")

            except queue.Empty:
                continue
                def run(self):
        """Start camera and control threads."""
        self.running = True

        cam_thread = Thread(target=self.camera_thread, daemon=True)
        ctrl_thread = Thread(target=self.control_thread, daemon=True)

        cam_thread.start()
        ctrl_thread.start()

        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            self.running = False

        cam_thread.join()
        ctrl_thread.join()
        print("[INFO] PID Controller stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_vis", action="store_true", help="Disable camera visualization")
    controller = PIDController(parser.parse_args())
    controller.run()