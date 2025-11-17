"""PID control script for beam balancing via HSV tracking and stepper control.

This module:
- Captures frames from a camera.
- Detects the ball, tip, and center markers in HSV color space.
- Computes the angular error between the ball and tip vectors about the center.
- Runs a PID loop (with trackbars to tune Kp/Ki/Kd live) to rotate a stepper motor
  until the error is within a deadband.
"""

import math
import time

import numpy as np


class PIDController:
    """PID controller for beam balancing using servo control."""

    def __init__(self, Kp=0, Ki=0, Kd=0, scale_factor=0.25):
        """Initialize controller and PID parameters."""
        # PID gains
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd

        self.integral = 0.0
        self.prev_error = 0.0
        self.last_time = None
        self.scale_factor = scale_factor

    def reset(self):
        """Reset PID controller state."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.last_time = None

    def update_pid(
        self,
        setpoint,
        position,
        dt=None,
        vel_control=False,
        p_scaler=1,
        d_scaler=1,
        deadband=0.012,
        integral_term=True,
    ):
        """Perform PID calculation and return control output."""
        error = setpoint - position
        if abs(error) < deadband:
            error = 0
        error = error * self.scale_factor  # Scale error for easier tuning (if needed)

        current_time = time.time()

        # Compute time difference
        if self.last_time is None:
            dt = 0.0
        else:
            dt = current_time - self.last_time

        # Proportional term
        P_val = self.Kp * error * p_scaler

        # Integral term accumulation
        if dt > 0 and integral_term:
            self.integral += error * dt
        I_val = self.Ki * self.integral

        # Derivative term calculation
        derivative = (error - self.prev_error) / dt if dt > 0 else 0

        D_val = self.Kd * d_scaler * derivative

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
        self.pi_x = PIDController(Kp=25.6, Ki=5.0)
        self.pi_y = PIDController(Kp=25.6, Ki=5.0)
        # Middle loop (velocity PD)
        self.pd_vx = PIDController(Kp=1.2, Kd=1.8)
        self.pd_vy = PIDController(Kp=1.2, Kd=1.8)

        # Inner loop (plate PID)
        self.pid_pitch = PIDController(Kp=0.45, Ki=0.25, Kd=0.0)
        self.pid_roll = PIDController(Kp=0.45, Ki=0.25, Kd=0.0)

        # Targets
        self.ball_setpoint = np.array([0.00, 0.0])
        self.vel_setpoint = np.array([0.00, 0.0])
        self.z_setpoint = 0.0
        self.time = 0.0

        # Linear acceleration to angle mapping
        self.g = 9.81
        self.a_to_angle_factor = (3.0 / 5.0) * self.g
        # Target ball position (x, y)

    def update_control(self, ball_kinematics, plate_pitch, plate_roll, dt=0.03):
        """Main PID control loop for Stewart Platform."""
        # Predict + update position and velocity

        est_x, est_y, est_vx, est_vy = ball_kinematics
        est_x = est_x + est_vx * dt
        est_y = est_y + est_vy * dt

        # Outer loop (Position PI)
        pos_scaler_term = 1
        if (
            math.sqrt(
                (est_x - self.ball_setpoint[0]) ** 2
                + (est_y - self.ball_setpoint[1]) ** 2
            )
            < 0.03
        ):
            pos_scaler_term = 0.5
        vx_set = self.pi_x.update_pid(
            self.ball_setpoint[0], est_x, dt, p_scaler=pos_scaler_term
        )
        vy_set = self.pi_y.update_pid(
            self.ball_setpoint[1], est_y, dt, p_scaler=pos_scaler_term
        )

        # Middle loop (Velocity PD)
        roll_acc = 0
        pitch_acc = 0
        if (
            math.sqrt(
                (est_vx - self.vel_setpoint[0]) ** 2
                + (est_vy - self.vel_setpoint[1]) ** 2
            )
            < 0.03
        ):
            roll_acc = self.pd_vx.update_pid(
                self.vel_setpoint[0] + vx_set, est_vx, dt, d_scaler=0.2
            )
            pitch_acc = self.pd_vy.update_pid(
                self.vel_setpoint[1] + vy_set, est_vy, dt, d_scaler=0.2
            )
        else:
            roll_acc = self.pd_vx.update_pid(self.vel_setpoint[0] + vx_set, est_vx, dt)
            pitch_acc = self.pd_vy.update_pid(self.vel_setpoint[1] + vy_set, est_vy, dt)

        # Map acceleration to linear angles
        pitch_des = np.degrees(
            np.arcsin(np.clip(pitch_acc / self.a_to_angle_factor, -1, 1))
        )
        roll_des = np.degrees(
            np.arcsin(np.clip(roll_acc / self.a_to_angle_factor, -1, 1))
        )

        print(f"pitch (2D): {abs(pitch_des):.3f}, ")
        # Inner loop (Plate orientation PID)
        # integral_plate = True
        # if (
        #     math.sqrt(est_x**2 + est_y**2) < 0.012
        #     or math.sqrt(est_x**2 + est_y**2) > 0.10
        # ):
        #     integral_plate = False
        # pitch_out = self.pid_pitch.update_pid(
        #     pitch_des, plate_pitch, dt, deadband=0.5, integral_term=integral_plate
        # )
        # roll_out = self.pid_roll.update_pid(
        #     roll_des, plate_roll, dt, deadband=0.5, integral_term=integral_plate
        # )

        # Z-axis offset
        z_out = self.z_setpoint
        # self.time += dt*8
        # self.ball_setpoint[0] = math.cos(self.time)*0.04
        # self.ball_setpoint[1] = math.sin(self.time)*0.04
        # print(self.ball_setpoint[0])

        return pitch_des, roll_des, z_out
