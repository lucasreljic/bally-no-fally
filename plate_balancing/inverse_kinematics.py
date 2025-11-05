"""Inverse Kinematics for Stewart Platform."""
import math

import numpy as np

# from servo_control import set_angle


class InverseKinematics:
    """Inverse Kinematics class for Stewart Platform."""

    def __init__(
        self,
        platform_radius=80,
        first_link_length=52,
        second_link_length=87,
        base_radius=100,
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
            x = self.base_radius * math.cos(angle)  # Radial position
            y = self.base_radius * math.sin(angle)
            self.servo_positions.append(np.array([x, y, 0]))

        # Calculate platform connection points
        self.platform_joints = self._calculate_platform_joints()

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
        roll = math.radians(roll)

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

            # Use law of cosines to solve for first link angle
            target_distance = np.linalg.norm(target_vector)

            # Check if target is reachable
            if target_distance > (self.l1 + self.l2):
                raise ValueError(f"Target position unreachable for servo {i}")

            cos_theta = (self.l1**2 + target_distance**2 - self.l2**2) / (
                2 * self.l1 * target_distance
            )
            cos_theta = np.clip(cos_theta, -1.0, 1.0)  # Prevent domain errors

            # Calculate servo angle in servo's local coordinate frame
            theta = math.acos(cos_theta)

            # Project onto servo's radial plane and calculate final angle
            xy_dist = math.sqrt(target_vector[0] ** 2 + target_vector[1] ** 2)
            elevation = math.atan2(target_vector[2], xy_dist)

            # Combine angles and convert to degrees
            servo_angle = math.degrees(theta + elevation)

            # Add offset based on servo mounting angle
            servo_angle += math.degrees(self.servo_angles[i])

            servo_angles.append(servo_angle)

        return servo_angles

    def move_to_position(self, pitch, roll, z):
        """Move the platform to specified position and orientation.

        Returns: calculated servo angles
        """
        servo_angles = self.calculate_leg_angles(pitch, roll, z)

        # Set servo angles using servo_control.py
        for i, angle in enumerate(servo_angles):
            # servo angle here
            # set_angle(i, angle)

            continue

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


def main(roll=0, pitch=0, z=0):
    """Test Inverse Kinematics calculations."""
    ik = InverseKinematics()
    # servo_0 = ServoController(config_file="plate_balancing/servo_0.json")
    # servo_1 = ServoController(config_file="plate_balancing/servo_1.json")
    # servo_2 = ServoController(config_file="plate_balancing/servo_2.json")

    # Example target positions
    targets = [
        (roll, pitch, z),
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

    main(args)
