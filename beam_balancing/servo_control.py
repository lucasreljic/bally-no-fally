"""Servo Control Module for Ball and Beam System.

PWM-based servo motor control for tilting the beam based on ball position.
Supports both direct angle control and position-based control with configurable limits.
"""

import json
import os
import time

try:
    import RPi.GPIO as GPIO
except ImportError:
    print("[SERVO] Warning: RPi.GPIO not available, using mock mode")
    GPIO = None


class ServoController:
    """PWM servo motor controller for beam positioning."""

    def __init__(self, config_file="config.json"):
        """Initialize servo controller with configuration.

        Args:
            config_file (str): Path to JSON config file with servo settings
        """
        # Default servo configuration
        self.servo_pin = 18  # GPIO pin for servo control
        self.pwm_frequency = 50  # Standard servo frequency (50Hz)
        self.min_pulse_width = 1.0  # Minimum pulse width in ms
        self.max_pulse_width = 2.0  # Maximum pulse width in ms
        self.center_pulse_width = 1.5  # Center position pulse width in ms
        self.min_angle = -45  # Minimum servo angle in degrees
        self.max_angle = 45  # Maximum servo angle in degrees
        self.current_angle = 0.0  # Current servo position

        self.pwm = None
        self.gpio_initialized = False

        # Load configuration from file if it exists
        if os.path.exists(config_file):
            try:
                with open(config_file, "r") as f:
                    config = json.load(f)

                # Extract servo configuration
                if "servo" in config:
                    servo_config = config["servo"]
                    self.servo_pin = servo_config.get("pin", self.servo_pin)
                    self.pwm_frequency = servo_config.get(
                        "frequency", self.pwm_frequency
                    )
                    self.min_pulse_width = servo_config.get(
                        "min_pulse_width", self.min_pulse_width
                    )
                    self.max_pulse_width = servo_config.get(
                        "max_pulse_width", self.max_pulse_width
                    )
                    self.center_pulse_width = servo_config.get(
                        "center_pulse_width", self.center_pulse_width
                    )
                    self.min_angle = servo_config.get("min_angle", self.min_angle)
                    self.max_angle = servo_config.get("max_angle", self.max_angle)

                print(
                    f"[SERVO] Loaded config: Pin {self.servo_pin}, Range {self.min_angle}° to {self.max_angle}°"
                )

            except Exception as e:
                print(f"[SERVO] Config load error: {e}, using defaults")
        else:
            print("[SERVO] No config file found, using default servo settings")

        # Initialize GPIO and PWM
        self._initialize_gpio()

    def _initialize_gpio(self):
        """Initialize GPIO and PWM for servo control."""
        if GPIO is None:
            print("[SERVO] GPIO not available, running in mock mode")
            return

        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.servo_pin, GPIO.OUT)

            self.pwm = GPIO.PWM(self.servo_pin, self.pwm_frequency)
            self.pwm.start(0)  # Start PWM with 0% duty cycle

            # Move to center position
            self.set_angle(0)
            self.gpio_initialized = True

            print(f"[SERVO] GPIO initialized on pin {self.servo_pin}")

        except Exception as e:
            print(f"[SERVO] GPIO initialization error: {e}")

    def _angle_to_duty_cycle(self, angle):
        """Convert angle to PWM duty cycle.

        Args:
            angle (float): Servo angle in degrees

        Returns:
            float: PWM duty cycle percentage
        """
        # Clamp angle to valid range
        angle = max(self.min_angle, min(self.max_angle, angle))

        # Map angle to pulse width
        angle_range = self.max_angle - self.min_angle
        pulse_range = self.max_pulse_width - self.min_pulse_width

        # Calculate pulse width for given angle
        angle_normalized = (angle - self.min_angle) / angle_range
        pulse_width = self.min_pulse_width + (angle_normalized * pulse_range)

        # Convert pulse width to duty cycle (pulse_width in ms, period = 20ms for 50Hz)
        duty_cycle = (pulse_width / 20.0) * 100.0

        return duty_cycle

    def set_angle(self, angle):
        """Set servo to specific angle.

        Args:
            angle (float): Target angle in degrees (-45 to +45 typically)
        """
        # Clamp angle to configured range
        angle = max(self.min_angle, min(self.max_angle, angle))

        if self.pwm and self.gpio_initialized:
            try:
                duty_cycle = self._angle_to_duty_cycle(angle)
                self.pwm.ChangeDutyCycle(duty_cycle)
                self.current_angle = angle

                print(f"[SERVO] Set angle: {angle:.1f}° (duty: {duty_cycle:.1f}%)")

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
        """Clean up GPIO resources."""
        if self.pwm and self.gpio_initialized:
            try:
                self.pwm.stop()
                GPIO.cleanup()
                print("[SERVO] GPIO cleaned up")
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
