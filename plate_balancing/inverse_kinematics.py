"""Inverse Kinematics for Stewart Platform."""
import math
import time

import numpy as np
from servo_plate_control import ServoController


class InverseKinematics:
    """Inverse Kinematics class for Stewart Platform."""

    def __init__(
        self,
        platform_radius=150,
        first_link_length=80,
        second_link_length=95,
        base_radius=50,
    ):
        """Initialize Ball Balancing Platform parameters.

        platform_radius: radius of the moving platform (mm)
        first_link_length: length of servo-driven link (mm)
        second_link_length: length of intermediate link (mm)
        base_radius: radius of the circle on which servos are mounted (mm)
        """
        self.platform_radius = platform_radius
        self.l1 = first_link_length
        self.l2 = second_link_length
        self.base_radius = base_radius

        # Platform joint angles (120° spacing, aligned with first servo)
        self.servo_angles = [0, 120, 240]  # First servo aligned with x-axis
        self.platform_angles = self.servo_angles  # Platform joints aligned with servos
        self.servo_angles = [math.radians(angle) for angle in self.servo_angles]
        self.platform_angles = [math.radians(angle) for angle in self.platform_angles]

        # Calculate servo mounting positions in 3D space
        self.servo_positions = []
        for angle in self.servo_angles:
            x = self.base_radius * math.cos(angle)
            y = self.base_radius * math.sin(angle)
            self.servo_positions.append(np.array([x, y, 0]))

        # Calculate platform connection points
        self.platform_joints = self._calculate_platform_joints()

        # Initialize Servo Controllers
        self.servos = [
            ServoController(config_file=f"plate_balancing/servo_{i}.json")
            for i in range(3)
        ]

    def _calculate_platform_joints(self):
        """Calculate the x, y coordinates of platform connection points."""
        positions = []
        for angle in self.platform_angles:
            x = self.platform_radius * math.cos(angle)
            y = self.platform_radius * math.sin(angle)
            positions.append(np.array([x, y, 0]))
        return positions

    def calculate_leg_angles(self, pitch, roll, z):
        """Calculate required servo angles for given pitch, roll, and z position.

        pitch: rotation around x-axis (degrees)
        roll: rotation around y-axis (degrees)
        z: vertical displacement (mm)
        Returns: list of 3 servo angles in degrees
        """
        # Convert angles to radians
        pitch = math.radians(pitch)
        roll = np.deg2rad(roll)

        # Create rotation matrices
        Rx = np.array(
            [
                [1, 0, 0],
                [0, math.cos(pitch), -math.sin(pitch)],
                [0, math.sin(pitch), math.cos(pitch)],
            ]
        )

        Ry = np.array(
            [
                [math.cos(roll), 0, math.sin(roll)],
                [0, 1, 0],
                [-math.sin(roll), 0, math.cos(roll)],
            ]
        )

        R = Rx @ Ry
        T = np.array([0, 0, z])

        servo_angles = []
        for i in range(3):
            # Transform platform joint position
            platform_pos = R @ self.platform_joints[i] + T

            # Vector from servo base to platform connection
            target_vector = platform_pos - self.servo_positions[i]

            # Calculate required leg length for 2-link system
            target_distance = np.linalg.norm(target_vector)

            # Check if target is reachable
            if target_distance > (self.l1 + self.l2):
                raise ValueError(f"Target position unreachable for servo {i}")
            if target_distance < abs(self.l1 - self.l2):
                raise ValueError(f"Target position too close for servo {i}")

            # Use law of cosines to find angle between first link and target vector
            cos_theta = (self.l1**2 + target_distance**2 - self.l2**2) / (
                2 * self.l1 * target_distance
            )
            cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Prevent domain errors
            theta = math.acos(cos_theta)

            # Calculate elevation angle from horizontal to target
            horizontal_dist = math.sqrt(target_vector[0] ** 2 + target_vector[1] ** 2)
            elevation = math.atan2(target_vector[2], horizontal_dist)

            # Calculate reference angle (angle when platform is at [0,0,0])
            ref_vector = self.platform_joints[i] - self.servo_positions[i]
            ref_distance = np.linalg.norm(ref_vector)
            ref_cos_theta = (self.l1**2 + ref_distance**2 - self.l2**2) / (
                2 * self.l1 * ref_distance
            )
            ref_cos_theta = np.clip(ref_cos_theta, -1.0, 1.0)
            ref_theta = math.acos(ref_cos_theta)
            ref_horizontal_dist = math.sqrt(ref_vector[0] ** 2 + ref_vector[1] ** 2)
            ref_elevation = math.atan2(ref_vector[2], ref_horizontal_dist)
            ref_angle = ref_elevation + ref_theta - math.pi / 2

            # Servo angle: positive moves up, negative moves down, zero at reference
            servo_angle = math.degrees(elevation + theta - math.pi / 2 - ref_angle)

            servo_angles.append(servo_angle)

        return servo_angles

    def move_to_position(self, pitch, roll, z):
        """Move the platform to specified position and orientation.

        Returns: calculated servo angles
        """
        servo_angles = self.calculate_leg_angles(pitch, roll, z)

        # Set servo angles using servo_control.py
        for i, angle in enumerate(servo_angles):
            self.servos[i].set_angle(angle)

        return servo_angles

    def get_platform_state(self, servo_angles):
        """Calculate platform position and orientation from servo angles.

        Useful for PID controller feedback
        servo_angles: list of 3 servo angles in degrees
        Returns: (pitch, roll, z) tuple
        """
        # Implementation would go here - this would be used by the PID controller
        # to determine current platform state
        pass


def main(pitch=0, roll=0, z=0):
    """Test Inverse Kinematics calculations."""
    ik = InverseKinematics()

    # Example target positions
    targets = [
        (pitch, roll, z),
        (-5, 5, 5),
        (5, -5, 5),
        (-5, -5, 5),
        (0, 0, 0),
    ]

    for pitch, roll, z in targets:
        try:
            angles = ik.calculate_leg_angles(pitch, roll, z)
            print(
                f"Target Pitch: {pitch}°, Roll: {roll}°, Z: {z}mm -> Servo Angles: {angles}"
            )
            ik.move_to_position(pitch, roll, z)
            time.sleep(1)  # Pause to observe movement
        except ValueError as e:
            print(e)


if __name__ == "__main__":
    """
    Entry point for Inverse Kinematics test.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Test Inverse Kinematics for Stewart Platform"
    )
    parser.add_argument(
        "--pitch", type=float, default=0, help="Platform pitch (degrees)"
    )
    parser.add_argument("--roll", type=float, default=0, help="Platform roll (degrees)")
    parser.add_argument("--z", type=float, default=0, help="Platform z position (mm)")
    args = parser.parse_args()

    main(args.pitch, args.roll, args.z)
