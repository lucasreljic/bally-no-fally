"""PID control script for beam balancing via HSV tracking and stepper control.

This module:
- Captures frames from a camera.
- Detects the ball, tip, and center markers in HSV color space.
- Computes the angular error between the ball and tip vectors about the center.
- Runs a PID loop (with trackbars to tune Kp/Ki/Kd live) to rotate a stepper motor
  until the error is within a deadband.
"""

import time

import numpy as np


class PIDController:
    """PID controller for beam balancing using servo control."""

    def __init__(self, args, Kp=0, Ki=0, Kd=0, scale_factor=0.25):
        """Initialize controller and PID parameters."""
        # PID gains
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.integral = 0.0
        self.prev_error = 0.0
        self.last_time = None

    def reset(self):
        """Reset PID controller state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_time = None

    def update_pid(self, setpoint, position, dt=None):
        """Perform PID calculation and return control output."""
        error = setpoint - position
        current_time = time.time()

        # Compute time difference
        if self.last_time is None:
            dt = 0.0
        else:
            dt = current_time - self.last_time

        # Proportional term
        P_val = self.Kp * error

        # Integral term accumulation
        if dt > 0:
            self.integral += error * dt
        I_val = self.Ki * self.integral

        # Derivative term calculation
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        D_val = self.Kd * derivative

        # PID output (limit to safe beam range)
        output = P_val + I_val + D_val
        output = np.clip(output, -15, 15)

        self.prev_error = error
        self.last_time = current_time

        return output


class StewartPlatformController:
    """Multi axis controller for Stewart Platform.

    - PI for ball position
    - PD for ball velocity
    - PID for platform orientation (pitch/roll)
    """

    def __init__(self, scale_factor=0.25):
        """Initialize Stewart Platform Controller."""
        # Outer loop (position PI)
        self.pi_x = PIDController(Kp=0.6, Ki=0.2)
        self.pi_y = PIDController(Kp=0.6, Ki=0.2)

        # Middle loop (velocity PD)
        self.pd_vx = PIDController(Kp=0.5, Kd=0.1)
        self.pd_vy = PIDController(Kp=0.5, Kd=0.1)

        # Inner loop (plate PID)
        self.pid_pitch = PIDController(Kp=1.0, Ki=0.1, Kd=0.05)
        self.pid_roll = PIDController(Kp=1.0, Ki=0.1, Kd=0.05)

        # Targets
        self.ball_setpoint = np.array([0.0, 0.0])
        self.vel_setpoint = np.array([0.0, 0.0])
        self.z_setpoint = 0.0

        # Linear acceleration to angle mapping
        self.g = 9.81
        self.a_to_angle_factor = (5.0 / 7.0) * self.g
        # Target ball position (x, y)

    def update_control(self, ball_kinematics, plate_pitch, plate_roll, dt=0.02):
        """Main PID control loop for Stewart Platform."""
        # Predict + update position and velocity

        est_x, est_y, est_vx, est_vy = ball_kinematics
        # Outer loop (Position PI)
        vx_set, _ = self.pi_x.update_pid(self.ball_setpoint[0], est_x, dt)
        vy_set, _ = self.pi_y.update_pid(self.ball_setpoint[1], est_y, dt)

        # Middle loop (Velocity PD)
        pitch_acc, _ = self.pd_vx.update_pid(self.vel_setpoint[0] + vx_set, est_vx, dt)
        roll_acc, _ = self.pd_vy.update_pid(self.vel_setpoint[1] + vy_set, est_vy, dt)

        # Map acceleration to linear angles
        pitch_des = np.degrees(np.arcsin(np.clip(pitch_acc / self.a_to_angle_factor)))
        roll_des = np.degrees(np.arcsin(np.clip(roll_acc / self.a_to_angle_factor)))

        # Inner loop (Plate orientation PID)
        pitch_out, _ = self.pid_pitch.update_pid(pitch_des, plate_pitch, dt)
        roll_out, _ = self.pid_roll.update_pid(roll_des, plate_roll, dt)

        # Z-axis offset
        z_out = self.z_setpoint

        return pitch_out, roll_out, z_out
