"""Servo Control Module for Ball and Beam System.

PWM-based servo motor control for tilting the beam based on ball position.
Supports both direct angle control and position-based control with configurable limits.
Uses PCA9685 hardware PWM controller to eliminate jitter.
"""

import json
import os
import time

# Try multiple PWM backends in order of preference
pwm_backend = None
pin_factory = None

# First try PCA9685 hardware PWM controller (best option - no jitter)
try:
    import smbus

    pwm_backend = "pca9685"
    print("[SERVO] Using PCA9685 hardware PWM controller (no jitter)")
except ImportError:
    pass

# Fallback to gpiozero options if PCA9685 not available
if pwm_backend is None:
    try:
        from gpiozero import Servo
        from gpiozero.pins.lgpio import LGPIOFactory

        pin_factory = LGPIOFactory()
        pwm_backend = "gpiozero-lgpio"
        print("[SERVO] Using gpiozero with lgpio pin factory (RPi5 compatible)")
    except ImportError:
        pass


class PCA9685:
    """PCA9685 16-channel PWM controller driver."""

    def __init__(self, address=0x40, busnum=1):
        """Initialize PCA9685 PWM controller.

        Args:
            address (int): I2C address of PCA9685 (default 0x40)
            busnum (int): I2C bus number (default 1)
        """
        self.bus = smbus.SMBus(busnum)
        self.address = address
        self._setup()

    def _setup(self):
        """Set up PCA9685 with default configuration."""
        self.write(0x00, 0x00)  # Mode1 register - normal mode
        self.set_pwm_freq(50)  # Set frequency to 50 Hz for servos

    def write(self, reg, value):
        """Write byte to PCA9685 register."""
        self.bus.write_byte_data(self.address, reg, value)

    def read(self, reg):
        """Read byte from PCA9685 register."""
        return self.bus.read_byte_data(self.address, reg)

    def set_pwm_freq(self, freq_hz):
        """Set PWM frequency for all channels.

        Args:
            freq_hz (float): Frequency in Hz (typical servo frequency is 50Hz)
        """
        prescale_val = int(25000000.0 / (4096 * freq_hz) - 1)
        old_mode = self.read(0x00)
        new_mode = (old_mode & 0x7F) | 0x10  # Sleep mode
        self.write(0x00, new_mode)  # Go to sleep
        self.write(0xFE, prescale_val)  # Set prescaler
        self.write(0x00, old_mode)  # Restore mode
        time.sleep(0.005)
        self.write(0x00, old_mode | 0x80)  # Wake up with auto-increment

    def set_pwm(self, channel, on, off):
        """Set PWM for specific channel.

        Args:
            channel (int): PWM channel (0-15)
            on (int): ON time (0-4095)
            off (int): OFF time (0-4095)
        """
        self.write(0x06 + 4 * channel, on & 0xFF)
        self.write(0x07 + 4 * channel, on >> 8)
        self.write(0x08 + 4 * channel, off & 0xFF)
        self.write(0x09 + 4 * channel, off >> 8)

    def set_pulse_width(self, channel, pulse_width_ms, frequency=50):
        """Set servo pulse width in milliseconds.

        Args:
            channel (int): PWM channel (0-15)
            pulse_width_ms (float): Pulse width in milliseconds
            frequency (float): PWM frequency in Hz
        """
        pulse_length = (
            1000000.0 / frequency / 4096
        )  # Length of one pulse in microseconds
        pulse = int(pulse_width_ms * 1000 / pulse_length)  # Convert to PWM value
        self.set_pwm(channel, 0, pulse)


class PCA9685Servo:
    """Servo controller using PCA9685 PWM driver."""

    def __init__(self, pca9685, channel, min_pulse_width=1.0, max_pulse_width=2.0):
        """Initialize servo on PCA9685 channel.

        Args:
            pca9685 (PCA9685): PCA9685 instance
            channel (int): PWM channel (0-15)
            min_pulse_width (float): Minimum pulse width in ms
            max_pulse_width (float): Maximum pulse width in ms
        """
        self.pca9685 = pca9685
        self.channel = channel
        self.min_pulse_width = min_pulse_width
        self.max_pulse_width = max_pulse_width

    def set_angle(self, angle, min_angle=-90, max_angle=90):
        """Set servo to specific angle.

        Args:
            angle (float): Target angle in degrees
            min_angle (float): Minimum angle range
            max_angle (float): Maximum angle range
        """
        # Clamp angle to range
        angle = max(min_angle, min(max_angle, angle))

        # Map angle to pulse width
        angle_range = max_angle - min_angle
        pulse_range = self.max_pulse_width - self.min_pulse_width

        normalized = (angle - min_angle) / angle_range  # 0 to 1
        pulse_width = self.min_pulse_width + (normalized * pulse_range)

        # Set PWM pulse width
        self.pca9685.set_pulse_width(self.channel, pulse_width)

    def set_pulse_width(self, pulse_width_ms):
        """Set servo position by pulse width."""
        self.pca9685.set_pulse_width(self.channel, pulse_width_ms)

    def smooth_move(
        self, start_angle, end_angle, step=0.1, delay=0.05, min_angle=-90, max_angle=90
    ):
        """Smoothly move servo from start to end angle.

        Args:
            start_angle (float): Starting angle
            end_angle (float): Ending angle
            step (float): Step size in degrees
            delay (float): Delay between steps in seconds
            min_angle (float): Minimum angle range
            max_angle (float): Maximum angle range
        """
        if start_angle < end_angle:
            angle = start_angle
            while angle <= end_angle:
                self.set_angle(angle, min_angle, max_angle)
                angle += step
                time.sleep(delay)
        else:
            angle = start_angle
            while angle >= end_angle:
                self.set_angle(angle, min_angle, max_angle)
                angle -= step
                time.sleep(delay)


class ServoController:
    """PWM servo motor controller for beam positioning."""

    def __init__(self, config_file="config.json"):
        """Initialize servo controller with configuration.

        Args:
            config_file (str): Path to JSON config file with servo settings
        """
        # Default servo configuration
        self.servo_pin = 18  # GPIO pin for servo control
        self.safety_min_angle = -6  # For the Beam balancer this is the max travel angle
        self.safety_max_angle = 6
        self.min_angle = -46  # Minimum servo angle in degrees
        self.max_angle = 44  # Maximum servo angle in degrees
        self.current_angle = 0.0  # Current servo position

        # Servo PWM parameters
        self.min_pulse_width = 1.0  # 1ms in milliseconds
        self.max_pulse_width = 2.0  # 2ms in milliseconds
        self.pwm_frequency = 50  # 50Hz standard for servos

        # PCA9685 specific configuration
        self.pca9685_address = 0x40  # I2C address
        self.pca9685_channel = 0  # PWM channel on PCA9685
        self.pca9685_busnum = 1  # I2C bus number

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
                    self.min_pulse_width = servo_config.get(
                        "min_pulse_width", self.min_pulse_width
                    )
                    self.max_pulse_width = servo_config.get(
                        "max_pulse_width", self.max_pulse_width
                    )
                    self.pwm_frequency = servo_config.get(
                        "frequency", self.pwm_frequency
                    )

                    # PCA9685 specific config
                    if "pca9685" in servo_config:
                        pca_config = servo_config["pca9685"]
                        self.pca9685_address = pca_config.get(
                            "address", self.pca9685_address
                        )
                        self.pca9685_channel = pca_config.get(
                            "channel", self.pca9685_channel
                        )
                        self.pca9685_busnum = pca_config.get(
                            "busnum", self.pca9685_busnum
                        )

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
        """Initialize servo using best available PWM method."""
        # Try PCA9685 hardware PWM first (best option)
        if pwm_backend == "pca9685":
            try:
                self.pca9685 = PCA9685(self.pca9685_address, self.pca9685_busnum)
                self.pca9685.set_pwm_freq(self.pwm_frequency)

                self.servo = PCA9685Servo(
                    self.pca9685,
                    self.pca9685_channel,
                    self.min_pulse_width,
                    self.max_pulse_width,
                )

                self.set_angle(0)
                self.servo_initialized = True
                print(
                    f"[SERVO] PCA9685 servo initialized on channel {self.pca9685_channel} (address 0x{self.pca9685_address:02x})"
                )
                return
            except Exception as e:
                print(f"[SERVO] PCA9685 initialization failed: {e}")

        # Fallback to gpiozero methods
        if pwm_backend in ["gpiozero-lgpio", "pigpio", "gpiozero"]:
            try:
                if pin_factory:
                    self.servo = Servo(
                        self.servo_pin,
                        min_pulse_width=self.min_pulse_width / 1000.0,
                        max_pulse_width=self.max_pulse_width / 1000.0,
                        pin_factory=pin_factory,
                    )
                else:
                    self.servo = Servo(
                        self.servo_pin,
                        min_pulse_width=self.min_pulse_width / 1000.0,
                        max_pulse_width=self.max_pulse_width / 1000.0,
                    )

                self.set_angle(0)
                self.servo_initialized = True
                backend_name = (
                    "gpiozero with lgpio"
                    if pwm_backend == "gpiozero-lgpio"
                    else pwm_backend
                )
                print(
                    f"[SERVO] Servo initialized on pin {self.servo_pin} using {backend_name}"
                )
                return

            except Exception as e:
                print(f"[SERVO] gpiozero initialization error: {e}")

        print("[SERVO] No PWM backend available, running in mock mode")

    def _angle_to_servo_value(self, angle):
        """Convert angle to gpiozero servo value (-1 to +1).

        Args:
            angle (float): Servo angle in degrees

        Returns:
            float: Servo value for gpiozero (-1 to +1)
        """
        # Clamp angle to valid range
        angle = max(self.min_angle, min(self.max_angle, angle))
        angle = max(self.safety_min_angle, min(self.safety_max_angle, angle))
        # Map angle to servo value (-1 to +1)
        angle_range = self.max_angle - self.min_angle
        normalized = (angle - self.min_angle) / angle_range  # 0 to 1
        servo_value = (normalized * 2.0) - 1.0  # -1 to +1

        return servo_value

    def set_angle(self, angle):
        """Set servo to specific angle.

        Args:
            angle (float): Target angle in degrees
        """
        # Clamp angle to configured range
        angle = max(self.min_angle, min(self.max_angle, angle))
        # Apply safety limits
        angle = max(self.safety_min_angle, min(self.safety_max_angle, angle))

        if self.servo and self.servo_initialized:
            try:
                if isinstance(self.servo, PCA9685Servo):
                    # Use PCA9685 servo
                    self.servo.set_angle(angle, self.min_angle, self.max_angle)
                else:
                    # Use gpiozero servo
                    servo_value = self._angle_to_servo_value(angle)
                    self.servo.value = servo_value

                self.current_angle = angle
                print(f"[SERVO] Set angle: {angle:.1f}°")

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
        target_angle = -position * self.safety_max_angle

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
                if isinstance(self.servo, PCA9685Servo):
                    # Set servo to neutral position and stop PWM
                    self.servo.set_angle(0, self.min_angle, self.max_angle)
                    print("[SERVO] PCA9685 servo cleaned up")
                else:
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
    print("Commands: angle <degrees>, pos <-1.0 to 1.0>, center, sweep, quit")

    servo_controller = ServoController()

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
                    servo_controller.set_angle(angle)
                except ValueError:
                    print("Invalid angle value")
            elif cmd[0] == "pos" and len(cmd) > 1:
                try:
                    position = float(cmd[1])
                    servo_controller.set_position_normalized(position)
                except ValueError:
                    print("Invalid position value")
            elif cmd[0] == "center":
                servo_controller.center()
            elif cmd[0] == "sweep":
                print("Sweeping servo...")
                for angle in range(-30, 31, 5):
                    servo_controller.set_angle(angle)
                    time.sleep(0.5)
                servo_controller.center()
            else:
                print(
                    "Unknown command. Use: angle <deg>, pos <-1.0 to 1.0>, center, sweep, quit"
                )

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        servo_controller.cleanup()


if __name__ == "__main__":
    main()
