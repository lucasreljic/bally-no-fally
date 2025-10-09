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
from apriltags_detector import AprilTagDetector
from ball_detection import BallDetector
from servo_control import ServoController


class PIDController:
    """PID controller for beam balancing using servo control."""

    def __init__(
        self,
        args,
        servo_port="/dev/ttyUSB0",
        neutral_angle=11.4,
        kp=0.08,
        ki=0.04,
        kd=0.015,
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
        self.scale_error = 100
        # Beam length
        self.length_beam = 0.114  # meters
        self.deadband = 0.005  # meters
        self.I_range = 0.06 * self.scale_error  # meters
        # Scale factor for converting from pixels to meters
        self.scale_factor = scale_factor
        # Servo port name and center angle
        self.servo_port = servo_port
        self.neutral_angle = neutral_angle
        self.servo = None
        # Controller-internal state
        self.setpoint = 0.0
        self.integral = 0.0
        self.prev_error = 0.0
        # Data logs for plotting results (for debugging)
        self.time_log = []
        self.position_log = []
        self.setpoint_log = []
        self.control_log = []
        self.start_time = None
        # Thread-safe queue for most recent ball position measurement
        self.position_queue = queue.Queue(maxsize=1)
        self.running = False  # Main run flag for clean shutdown
        self.servo = None

    def connect_servo(self):
        """Try to open serial connection to servo, return True if success."""
        try:
            self.servo = serial.Serial(self.servo_port, 9600, timeout=0.5)
            time.sleep(1.5)
            print("[SERVO] Connected")
            return True
        except Exception as e:
            print(f"[SERVO] Connection Failed: {e}")
            self.servo = None
            return False

    def set_servo_pos(self, pos):
        """Set servo angle based on normalized position (-1 to 1)."""
        if not self.servo:
            return
        position = np.clip(pos, -1, 1)

        self.servo.set_position_normalized(position)
        return

    def send_servo_angle(self, angle):
        """Send angle command to servo motor (clipped for safety)."""
        if not self.servo:
            return
        servo_angle = int(np.clip(self.neutral_angle + angle, 0, 30))
        try:
            self.servo.write(bytes([servo_angle]))
        except Exception:
            print("[SERVO] Send failed")

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
        """Runs PID control loop in parallel with GUI and camera."""
        # if not self.connect_servo():
        #     print("[ERROR] No servo - running in simulation mode")
        self.servo = ServoController()
        self.start_time = time.time()
        while self.running:
            try:
                # Wait for latest ball position from camera
                position = self.position_queue.get(timeout=0.03)
                # Compute control output using PID
                control_output = self.update_pid(position)
                # Send control command to servo (real or simulated)
                self.set_servo_pos(control_output)
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
        if self.servo:
            self.servo.set_position_normalized(0)
            self.servo.cleanup()
            # Return to neutral on exit
            # self.send_servo_angle(0)
            # self.servo.close()

    # def create_gui(self):
    #     """Build Tkinter GUI with large sliders and labeled controls."""

    #     self.root = tk.Tk()
    #     self.root.title("Basic PID Controller")
    #     self.root.geometry("520x400")

    #     # Title label
    #     ttk.Label(self.root, text="PID Gains", font=("Arial", 18, "bold")).pack(pady=10)

    #     # Kp slider
    #     ttk.Label(self.root, text="Kp (Proportional)", font=("Arial", 12)).pack()
    #     self.kp_var = tk.DoubleVar(value=self.Kp)
    #     kp_slider = ttk.Scale(self.root, from_=0, to=100, variable=self.kp_var,
    #                           orient=tk.HORIZONTAL, length=500)
    #     kp_slider.pack(pady=5)
    #     self.kp_label = ttk.Label(self.root, text=f"Kp: {self.Kp:.1f}", font=("Arial", 11))
    #     self.kp_label.pack()

    #     # Ki slider
    #     ttk.Label(self.root, text="Ki (Integral)", font=("Arial", 12)).pack()
    #     self.ki_var = tk.DoubleVar(value=self.Ki)
    #     ki_slider = ttk.Scale(self.root, from_=0, to=10, variable=self.ki_var,
    #                           orient=tk.HORIZONTAL, length=500)
    #     ki_slider.pack(pady=5)
    #     self.ki_label = ttk.Label(self.root, text=f"Ki: {self.Ki:.1f}", font=("Arial", 11))
    #     self.ki_label.pack()

    #     # Kd slider
    #     ttk.Label(self.root, text="Kd (Derivative)", font=("Arial", 12)).pack()
    #     self.kd_var = tk.DoubleVar(value=self.Kd)
    #     kd_slider = ttk.Scale(self.root, from_=0, to=20, variable=self.kd_var,
    #                           orient=tk.HORIZONTAL, length=500)
    #     kd_slider.pack(pady=5)
    #     self.kd_label = ttk.Label(self.root, text=f"Kd: {self.Kd:.1f}", font=("Arial", 11))
    #     self.kd_label.pack()

    #     # Setpoint slider
    #     ttk.Label(self.root, text="Setpoint (meters)", font=("Arial", 12)).pack()
    #     pos_min = self.config['calibration']['position_min_m']
    #     pos_max = self.config['calibration']['position_max_m']
    #     self.setpoint_var = tk.DoubleVar(value=self.setpoint)
    #     setpoint_slider = ttk.Scale(self.root, from_=pos_min, to=pos_max,
    #                                variable=self.setpoint_var,
    #                                orient=tk.HORIZONTAL, length=500)
    #     setpoint_slider.pack(pady=5)
    #     self.setpoint_label = ttk.Label(self.root, text=f"Setpoint: {self.setpoint:.3f}m", font=("Arial", 11))
    #     self.setpoint_label.pack()

    #     # Button group for actions
    #     button_frame = ttk.Frame(self.root)
    #     button_frame.pack(pady=20)
    #     ttk.Button(button_frame, text="Reset Integral",
    #                command=self.reset_integral).pack(side=tk.LEFT, padx=5)
    #     ttk.Button(button_frame, text="Plot Results",
    #                command=self.plot_results).pack(side=tk.LEFT, padx=5)
    #     ttk.Button(button_frame, text="Stop",
    #                command=self.stop).pack(side=tk.LEFT, padx=5)

    #     # Schedule periodic GUI update
    #     self.update_gui()

    # def update_gui(self):
    #     """Reflect latest values from sliders into program and update display."""
    #     if self.running:
    #         # PID parameters
    #         self.Kp = self.kp_var.get()
    #         self.Ki = self.ki_var.get()
    #         self.Kd = self.kd_var.get()
    #         self.setpoint = self.setpoint_var.get()
    #         # Update displayed values
    #         self.kp_label.config(text=f"Kp: {self.Kp:.1f}")
    #         self.ki_label.config(text=f"Ki: {self.Ki:.1f}")
    #         self.kd_label.config(text=f"Kd: {self.Kd:.1f}")
    #         self.setpoint_label.config(text=f"Setpoint: {self.setpoint:.3f}m")
    #         # Call again after 50 ms (if not stopped)
    #         self.root.after(50, self.update_gui)

    # def reset_integral(self):
    #     """Clear integral error in PID (button handler)."""
    #     self.integral = 0.0
    #     print("[RESET] Integral term reset")

    # def plot_results(self):
    #     """Show matplotlib plots of position and control logs."""
    #     if not self.time_log:
    #         print("[PLOT] No data to plot")
    #         return
    #     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    #     # Ball position trace
    #     ax1.plot(self.time_log, self.position_log, label="Ball Position", linewidth=2)
    #     ax1.plot(self.time_log, self.setpoint_log, label="Setpoint",
    #              linestyle="--", linewidth=2)
    #     ax1.set_ylabel("Position (m)")
    #     ax1.set_title(f"Basic PID Control (Kp={self.Kp:.1f}, Ki={self.Ki:.1f}, Kd={self.Kd:.1f})")
    #     ax1.legend()
    #     ax1.grid(True, alpha=0.3)
    #     # Control output trace
    #     ax2.plot(self.time_log, self.control_log, label="Control Output",
    #              color="orange", linewidth=2)
    #     ax2.set_xlabel("Time (s)")
    #     ax2.set_ylabel("Beam Angle (degrees)")
    #     ax2.legend()
    #     ax2.grid(True, alpha=0.3)
    #     plt.tight_layout()
    #     plt.show()

    def stop(self):
        """Stop everything and clean up threads and GUI."""
        self.running = False
        # Try to safely close all windows/resources
        try:
            self.servo.cleanup()
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
        controller = PIDController(parser.parse_args())
        controller.run()
    except FileNotFoundError:
        print("[ERROR] config.json not found. Run simple_autocal.py first.")
    except Exception as e:
        print(f"[ERROR] {e}")
