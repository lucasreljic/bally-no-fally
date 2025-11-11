"""PID control script for beam balancing via HSV tracking and stepper control.

This module:
- Captures frames from a camera.
- Detects the ball, tip, and center markers in HSV color space.
- Computes the angular error between the ball and tip vectors about the center.
- Runs a PID loop (with trackbars to tune Kp/Ki/Kd live) to rotate a stepper motor
  until the error is within a deadband.
"""

import argparse
import os
import queue
import sys
import time
from threading import Thread

import cv2 as cv
import numpy as np
from servo_plate_control import ServoController

try:
    from beam_balancing.apriltags_detector import AprilTagDetector
    from beam_balancing.ball_detection import BallDetector
except ImportError:
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from beam_balancing.apriltags_detector import AprilTagDetector
    from beam_balancing.ball_detection import BallDetector


class PIDController:
    """PID controller for beam balancing using servo control."""

    def __init__(
        self,
        args,
        servo_port="/dev/ttyUSB0",
        kp=0.04,
        ki=0.03,
        kd=0.01,
        min_dist=0.04,
        scale_factor=0.25,
    ):
        """Initialize controller, load config, set defaults and queues."""
        # Load experiment and hardware config from JSON file (Alternate way)
        # with open(config_file, 'r') as f:
        #     self.config = json.load(f)
        # PID gains (controlled by sliders in GUI)

        self.Kp = kp
        self.Ki = ki
        self.Kd = kd

        self.visualization = not args.no_vis
        self.servo_name = 5 - args.tag_num
        self.scale_error = 50
        # Beam length
        self.length_beam = 0.298  # meters
        self.deadband = 0.005  # meters
        self.I_range = 0.09 * self.scale_error  # meters
        # Scale factor for converting from pixels to meters
        self.scale_factor = scale_factor
        # Servo port name and center angle
        self.servo_port = servo_port
        self.servo = None
        # Controller-internal state
        self.setpoint = 0.0
        self.integral = 0.0
        self.prev_error = 0.0
        self.min_distance = min_dist
        # Data logs for plotting results (for debugging)
        self.time_log = []
        self.position_log = []
        self.setpoint_log = []
        self.control_log = []
        self.start_time = None
        # Thread-safe queue for most recent ball position measurement
        self.position_queue = queue.Queue(maxsize=1)
        self.running = False  # Main run flag for clean shutdown

    def set_servo_pos(self, pos):
        """Set servo angle based on normalized position (-1 to 1)."""
        if not self.servo:
            return
        position = np.clip(pos, -1, 1)

        self.servo.set_position_normalized(position)
        return

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
        derivative = (error - self.prev_error) / dt
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
        detector = AprilTagDetector("camera_calibration.npz", tag_size_m=0.032)
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
            found = False
            pose_data = detector.detect_apriltag_poses(frame)
            tag_position = None
            required_tag = (
                5 - self.servo_name
            )  # or args.tag_num depending on what you want to track
            for pose_info in pose_data:
                if pose_info["tag_id"] == required_tag:
                    tag_position = pose_info
                    break

            if tag_position:
                vis_frame, found, _, distance_to_tag = ball_detector.draw_detection(
                    frame, apriltag_position=tag_position
                )

            if found and distance_to_tag:
                # Convert normalized to meters using scale
                position_m = (
                    distance_to_tag - self.min_distance
                ) - self.length_beam / 2
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
            if found and self.visualization:
                frame_with_overlay = detector.draw_detection_overlay(
                    vis_frame, pose_data
                )
                scale_percent = 50
                width = int(frame_with_overlay.shape[1] * scale_percent / 100)
                height = int(frame_with_overlay.shape[0] * scale_percent / 100)
                frame_with_overlay = cv.resize(frame_with_overlay, (width, height))
                cv.imshow("frame", frame_with_overlay)
            elif self.visualization:
                vis_frame, _, _, distance_to_tag = ball_detector.draw_detection(frame)
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
        """Runs PID control loop in parallel with GUI and camera."""
        # if not self.connect_servo():
        #     print("[ERROR] No servo - running in simulation mode")
        self.servo_arr = []
        if not self.servo:
            self.servo_arr.append(
                ServoController(config_file="plate_balancing/servo_0.json")
            )
            self.servo_arr.append(
                ServoController(config_file="plate_balancing/servo_1.json")
            )
            self.servo_arr.append(
                ServoController(config_file="plate_balancing/servo_2.json")
            )
            self.servo = self.servo_arr[self.servo_name]
            print(
                f"[SERVO] Controlled servo config: plate_balancing/servo_{str(self.servo_name)}.json"
            )
            self.servo_arr[0].set_position_normalized(0)
            self.servo_arr[1].set_position_normalized(0)
            self.servo_arr[2].set_position_normalized(0)

        self.start_time = time.time()
        while self.running:
            try:
                # Wait for latest ball position from camera
                position = self.position_queue.get(timeout=0.03)
                # Compute control output using PID
                control_output = self.update_pid(position)
                # Send control command to servo (real or simulated)
                self.set_servo_pos(-control_output)
                # Log results for plotting
                current_time = time.time() - self.start_time
                self.time_log.append(current_time)
                self.position_log.append(position)
                self.setpoint_log.append(self.setpoint)
                self.control_log.append(control_output)
                print(f"Pos: {position:.3f}m, Output: {control_output}")
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[CONTROL] Error: {e}")
                break
        if len(self.servo_arr) > 0:
            for i in self.servo_arr:
                self.servo_arr[i].set_position_normalized(0)

                self.servo_arr[i].cleanup()

            # Return to neutral on exit
            # self.servo.close()

    def stop(self):
        """Stop everything and clean up threads and GUI."""
        self.running = False
        # Try to safely close all windows/resources
        try:
            for i in self.servo_arr:
                self.servo_arr[i].cleanup()
            self.root.quit()
            self.root.destroy()
        except Exception:
            print("[ERROR] Unable to stop")

    def run(self):
        """Entry point: starts threads, launches GUI mainloop."""
        print("[INFO] Starting Basic PID Controller")

        self.running = True

        # Start camera and control threads, mark as daemon for exit
        cam_thread = Thread(target=self.camera_thread, daemon=True)
        ctrl_thread = Thread(target=self.control_thread, daemon=True)
        cam_thread.start()
        ctrl_thread.start()

        # Build and run GUI in main thread
        # self.create_gui()
        # self.root.mainloop()
        while True:
            time.sleep(1)
            if 0xFF == ord("q"):
                break
        # After GUI ends, stop everything
        cam_thread.join()
        ctrl_thread.join()
        self.running = False
        print("[INFO] Controller stopped")


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--no_vis", action="store_true", help="Disable Visualization"
        )
        parser.add_argument(
            "--tag_num",
            type=int,
            default=5,
            help="Tag number to track",
        )
        k_p = [0.04, 0.04, 0.04]
        k_i = [0.03, 0.03, 0.03]
        k_d = [0.01, 0.01, 0.01]
        min_distance = [0.03, 0.02, 0.02]

        num = 5 - parser.parse_args().tag_num
        controller = PIDController(
            parser.parse_args(),
            kp=k_p[num],
            ki=k_i[num],
            kd=k_d[num],
            min_dist=min_distance[num],
        )
        controller.run()
    except FileNotFoundError:
        print("[ERROR] config.json not found. Run simple_autocal.py first.")
    except Exception as e:
        print(f"[ERROR] {e}")
