"""Servo Control Module for Ball and Beam System.

PWM-based servo motor control for tilting the beam based on ball position.
Supports both direct angle control and position-based control with configurable limits.
"""

import json
import os
import time

try:
    from gpiozero import Servo
    from gpiozero.pins.pigpio import PiGPIOFactory

    # Use pigpio for more precise PWM timing
    try:
        pin_factory = PiGPIOFactory()
        print("[SERVO] Using pigpio pin factory for precise PWM")
    except Exception as e:
        pin_factory = None
        print(f"[SERVO] Using default pin factory: {e}")
except ImportError:
    print("[SERVO] Warning: gpiozero not available, using mock mode")
    Servo = None
    pin_factory = None


class ServoController:
    """PWM servo motor controller for beam positioning."""

    def __init__(self, config_file="config.json"):
        """Initialize servo controller with configuration.

        Args:
            config_file (str): Path to JSON config file with servo settings
        """
        # Default servo configuration
        self.servo_pin = 18  # GPIO pin for servo control
        self.min_angle = -45  # Minimum servo angle in degrees
        self.max_angle = 45  # Maximum servo angle in degrees
        self.current_angle = 0.0  # Current servo position

        # gpiozero servo parameters
        self.min_pulse_width = 1.0e-3  # 1ms in seconds
        self.max_pulse_width = 2.0e-3  # 2ms in seconds

        self.servo = None
        self.servo_initialized = False

        # Load configuration from file if it exists
        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    config = json.load(f)

                # Extract servo configuration
                if "servo" in config:
                    servo_config = config["servo"]
                    self.servo_pin = servo_config.get("pin", self.servo_pin)
                    self.min_angle = servo_config.get("min_angle", self.min_angle)
                    self.max_angle = servo_config.get("max_angle", self.max_angle)

                    # Convert pulse widths from ms to seconds for gpiozero
                    if "min_pulse_width" in servo_config:
                        self.min_pulse_width = servo_config["min_pulse_width"] / 1000.0
                    if "max_pulse_width" in servo_config:
                        self.max_pulse_width = servo_config["max_pulse_width"] / 1000.0

                print(
                    f"[SERVO] Loaded config: Pin {self.servo_pin}, Range {self.min_angle}° to {self.max_angle}°"
                )

            except Exception as e:
                print(f"[SERVO] Config load error: {e}, using defaults")
        else:
            print("[SERVO] No config file found, using default servo settings")

        # Initialize servo
        self._initialize_servo()

    def _initialize_servo(self):
        """Initialize servo using gpiozero."""
        if Servo is None:
            print("[SERVO] gpiozero not available, running in mock mode")
            return

        try:
            # Create servo object with custom pulse widths
            if pin_factory:
                self.servo = Servo(
                    self.servo_pin,
                    min_pulse_width=self.min_pulse_width,
                    max_pulse_width=self.max_pulse_width,
                    pin_factory=pin_factory,
                )
            else:
                self.servo = Servo(
                    self.servo_pin,
                    min_pulse_width=self.min_pulse_width,
                    max_pulse_width=self.max_pulse_width,
                )

            # Move to center position
            self.set_angle(0)
            self.servo_initialized = True

            print(f"[SERVO] Servo initialized on pin {self.servo_pin}")

        except Exception as e:
            print(f"[SERVO] Servo initialization error: {e}")

    def _angle_to_servo_value(self, angle):
        """Convert angle to gpiozero servo value (-1 to +1).

        Args:
            angle (float): Servo angle in degrees

        Returns:
            float: Servo value for gpiozero (-1 to +1)
        """
        # Clamp angle to valid range
        angle = max(self.min_angle, min(self.max_angle, angle))

        # Map angle to servo value (-1 to +1)
        angle_range = self.max_angle - self.min_angle
        normalized = (angle - self.min_angle) / angle_range  # 0 to 1
        servo_value = (normalized * 2.0) - 1.0  # -1 to +1

        return servo_value

    def set_angle(self, angle):
        """Set servo to specific angle.

        Args:
            angle (float): Target angle in degrees (-45 to +45 typically)
        """
        # Clamp angle to configured range
        angle = max(self.min_angle, min(self.max_angle, angle))

        if self.servo and self.servo_initialized:
            try:
                servo_value = self._angle_to_servo_value(angle)
                self.servo.value = servo_value
                self.current_angle = angle

                print(f"[SERVO] Set angle: {angle:.1f}° (value: {servo_value:.3f})")

            except Exception as e:
                print(f"[SERVO] Error setting angle: {e}")
        else:
            print(f"[SERVO] Mock mode - would set angle to {angle:.1f}°")
            self.current_angle = angle

    def set_position_normalized(self, position):
        """Set servo position based on normalized ball position.

        Args:
            position (float): Normalized position (-1.0 to +1.0)
                            -1.0 = ball far left, +1.0 = ball far right
        """
        # Map normalized position to servo angle
        # When ball is right (+1.0), tilt beam left (negative angle) to roll ball back
        # When ball is left (-1.0), tilt beam right (positive angle) to roll ball back
        target_angle = -position * self.max_angle

        self.set_angle(target_angle)

    def center(self):
        """Move servo to center position (0 degrees)."""
        self.set_angle(0)

    def get_current_angle(self):
        """Get current servo angle.

        Returns:
            float: Current servo angle in degrees
        """
        return self.current_angle

    def cleanup(self):
        """Clean up servo resources."""
        if self.servo and self.servo_initialized:
            try:
                self.servo.close()
                print("[SERVO] Servo cleaned up")
            except Exception as e:
                print(f"[SERVO] Cleanup error: {e}")

    def __del__(self):
        """Destructor - ensure cleanup on object deletion."""
        self.cleanup()


# Simple control functions for direct use
def create_servo_controller(config_file="config.json"):
    """Create and return a servo controller instance.

    Args:
        config_file (str): Path to configuration file

    Returns:
        ServoController: Configured servo controller instance
    """
    return ServoController(config_file)


# For testing when run directly
def main():
    """Test servo control with interactive commands."""
    print("Servo Control Test")
    print("Commands: angle <degrees>, pos <-1.0 to 1.0>, center, quit")

    servo = ServoController()

    try:
        while True:
            cmd = input("servo> ").strip().lower().split()

            if not cmd:
                continue

            if cmd[0] == "quit" or cmd[0] == "q":
                break
            elif cmd[0] == "angle" and len(cmd) > 1:
                try:
                    angle = float(cmd[1])
                    servo.set_angle(angle)
                except ValueError:
                    print("Invalid angle value")
            elif cmd[0] == "pos" and len(cmd) > 1:
                try:
                    position = float(cmd[1])
                    servo.set_position_normalized(position)
                except ValueError:
                    print("Invalid position value")
            elif cmd[0] == "center":
                servo.center()
            elif cmd[0] == "sweep":
                print("Sweeping servo...")
                for angle in range(-30, 31, 5):
                    servo.set_angle(angle)
                    time.sleep(0.5)
                servo.center()
            else:
                print(
                    "Unknown command. Use: angle <deg>, pos <-1.0 to 1.0>, center, sweep, quit"
                )

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        servo.cleanup()


if __name__ == "__main__":
    main()
