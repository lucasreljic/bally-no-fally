"""Servo Control Module for Ball and Beam System.

PWM-based servo motor control for tilting the beam based on ball position.
Supports both direct angle control and position-based control with configurable limits.
Uses hardware PWM when available to eliminate jitter on RPi5.
"""

import json
import os
import time

# Try multiple PWM backends in order of preference for RPi5
pwm_backend = None
pin_factory = None

# First try hardware PWM via sysfs (true hardware PWM for RPi5)
try:
    if os.path.exists("/sys/class/pwm/pwmchip0"):
        pwm_backend = "hardware"
        print("[SERVO] Using hardware PWM via sysfs (RPi5 - no jitter)")
    else:
        raise ImportError("Hardware PWM not available")
except ImportError:
    pass

# Fallback to rpi-lgpio
if pwm_backend is None:
    try:
        import lgpio

        pwm_backend = "lgpio"
        print("[SERVO] Using lgpio for PWM (RPi5 compatible)")
    except ImportError:
        pass

# Fallback to gpiozero with rpi-lgpio pin factory (RPi5 compatible)
if pwm_backend is None:
    try:
        from gpiozero import Servo
        from gpiozero.pins.lgpio import LGPIOFactory

        pin_factory = LGPIOFactory()
        pwm_backend = "gpiozero-lgpio"
        print("[SERVO] Using gpiozero with lgpio pin factory (RPi5 compatible)")
    except ImportError:
        pass

# Try gpiozero with pigpio (older Pi models)
if pwm_backend is None:
    try:
        from gpiozero import Servo
        from gpiozero.pins.pigpio import PiGPIOFactory

        try:
            pin_factory = PiGPIOFactory()
            pwm_backend = "pigpio"
            print("[SERVO] Using pigpio pin factory for precise PWM")
        except Exception as e:
            pwm_backend = "gpiozero"
            print(f"[SERVO] Using gpiozero software PWM (may have jitter): {e}")
    except ImportError:
        print("[SERVO] Warning: No PWM library available, using mock mode")


class HardwarePWMServo:
    """True hardware PWM servo controller using Linux PWM subsystem."""

    def __init__(self, pin, min_pulse_width=1.0, max_pulse_width=2.0, frequency=50):
        """Initialize hardware PWM servo.

        Args:
            pin (int): GPIO pin number for servo control
            min_pulse_width (float): Minimum pulse width in milliseconds (default 1.0ms)
            max_pulse_width (float): Maximum pulse width in milliseconds (default 2.0ms)
            frequency (int): PWM frequency in Hz (default 50Hz)
        """
        self.pin = pin
        self.min_pulse_width = min_pulse_width  # ms
        self.max_pulse_width = max_pulse_width  # ms
        self.frequency = frequency  # Hz
        self.current_duty_cycle = 0
        self.chip = None
        self.pwm_channel = None
        self.period_ns = None

        # GPIO pin to PWM channel mapping for RPi5
        # Only certain pins support hardware PWM
        self.pwm_channels = {
            12: (0, 0),  # PWM0 channel 0
            13: (0, 1),  # PWM0 channel 1
            18: (0, 0),  # PWM0 channel 0 (alternative)
            19: (0, 1),  # PWM0 channel 1 (alternative)
        }

        if pwm_backend == "hardware":
            self._init_hardware_pwm()
        elif pwm_backend == "lgpio":
            self._init_lgpio()
        else:
            raise RuntimeError("No hardware PWM backend available")

    def _init_hardware_pwm(self):
        """Initialize using Linux PWM subsystem for true hardware PWM."""
        if self.pin not in self.pwm_channels:
            raise RuntimeError(
                f"Pin {self.pin} does not support hardware PWM. Use pins: {list(self.pwm_channels.keys())}"
            )

        pwm_chip, pwm_channel = self.pwm_channels[self.pin]
        self.pwm_channel = pwm_channel

        pwm_path = f"/sys/class/pwm/pwmchip{pwm_chip}"
        channel_path = f"{pwm_path}/pwm{pwm_channel}"

        try:
            # Export the PWM channel if not already exported
            if not os.path.exists(channel_path):
                with open(f"{pwm_path}/export", "w") as f:
                    f.write(str(pwm_channel))

            # Set period (20ms = 20,000,000 ns for 50Hz)
            self.period_ns = int(1_000_000_000 / self.frequency)
            with open(f"{channel_path}/period", "w") as f:
                f.write(str(self.period_ns))

            # Set initial duty cycle to 0
            with open(f"{channel_path}/duty_cycle", "w") as f:
                f.write("0")

            # Enable PWM
            with open(f"{channel_path}/enable", "w") as f:
                f.write("1")

            self.channel_path = channel_path
            print(
                f"[SERVO] Hardware PWM initialized on pin {self.pin} (chip {pwm_chip}, channel {pwm_channel})"
            )

        except Exception as e:
            raise RuntimeError(f"Failed to initialize hardware PWM: {e}")

    def _init_lgpio(self):
        """Initialize using lgpio library as fallback."""
        self.chip = lgpio.gpiochip_open(0)
        lgpio.gpio_claim_output(self.chip, self.pin)
        lgpio.tx_pwm(self.chip, self.pin, self.frequency, 0)
        print(f"[SERVO] lgpio PWM initialized on pin {self.pin}")

    def set_pulse_width(self, pulse_width_ms):
        """Set servo position by pulse width in milliseconds."""
        if pwm_backend == "hardware":
            # Convert pulse width to duty cycle in nanoseconds
            duty_cycle_ns = int(pulse_width_ms * 1_000_000)  # ms to ns

            # Clamp to reasonable range
            min_duty_ns = int(0.5 * 1_000_000)  # 0.5ms
            max_duty_ns = int(2.5 * 1_000_000)  # 2.5ms
            duty_cycle_ns = max(min_duty_ns, min(max_duty_ns, duty_cycle_ns))

            try:
                with open(f"{self.channel_path}/duty_cycle", "w") as f:
                    f.write(str(duty_cycle_ns))
                self.current_duty_cycle = (duty_cycle_ns / self.period_ns) * 100
            except Exception as e:
                print(f"[SERVO] Error setting duty cycle: {e}")

        elif pwm_backend == "lgpio":
            # Convert pulse width to duty cycle percentage
            period_ms = 1000.0 / self.frequency
            duty_cycle = (pulse_width_ms / period_ms) * 100.0
            duty_cycle = max(0.5, min(12.5, duty_cycle))

            if self.chip is not None:
                import lgpio

                lgpio.tx_pwm(self.chip, self.pin, self.frequency, duty_cycle)
            self.current_duty_cycle = duty_cycle

    def set_angle(self, angle, min_angle=-90, max_angle=90):
        """Set servo angle (-90 to +90 degrees typically)."""
        # Clamp angle to range
        angle = max(min_angle, min(max_angle, angle))

        # Map angle to pulse width
        angle_range = max_angle - min_angle
        pulse_range = self.max_pulse_width - self.min_pulse_width

        normalized = (angle - min_angle) / angle_range  # 0 to 1
        pulse_width = self.min_pulse_width + (normalized * pulse_range)

        self.set_pulse_width(pulse_width)

    def cleanup(self):
        """Clean up PWM resources."""
        if pwm_backend == "hardware" and hasattr(self, "channel_path"):
            try:
                # Disable PWM
                with open(f"{self.channel_path}/enable", "w") as f:
                    f.write("0")

                # Unexport the channel
                pwm_chip = self.pwm_channels[self.pin][0]
                with open(f"/sys/class/pwm/pwmchip{pwm_chip}/unexport", "w") as f:
                    f.write(str(self.pwm_channel))

                print("[SERVO] Hardware PWM cleaned up")
            except Exception as e:
                print(f"[SERVO] Hardware PWM cleanup error: {e}")

        elif pwm_backend == "lgpio" and self.chip is not None:
            try:
                import lgpio

                lgpio.tx_pwm(self.chip, self.pin, self.frequency, 0)
                lgpio.gpio_free(self.chip, self.pin)
                lgpio.gpiochip_close(self.chip)
                self.chip = None
                print("[SERVO] lgpio PWM cleaned up")
            except Exception as e:
                print(f"[SERVO] lgpio cleanup error: {e}")


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

                print(
                    f"[SERVO] Loaded config: Pin {self.servo_pin}, Range {self.min_angle}Â° to {self.max_angle}Â°"
                )

            except Exception as e:
                print(f"[SERVO] Config load error: {e}, using defaults")
        else:
            print("[SERVO] No config file found, using default servo settings")

        # Initialize servo
        self._initialize_servo()

    def _initialize_servo(self):
        """Initialize servo using best available PWM method."""
        # Try hardware PWM first (true hardware PWM for RPi5)
        if pwm_backend in ["hardware", "lgpio"]:
            try:
                self.servo = HardwarePWMServo(
                    self.servo_pin,
                    self.min_pulse_width,
                    self.max_pulse_width,
                    self.pwm_frequency,
                )
                self.set_angle(0)
                self.servo_initialized = True
                backend_name = "hardware PWM" if pwm_backend == "hardware" else "lgpio"
                print(
                    f"[SERVO] {backend_name} servo initialized on pin {self.servo_pin}"
                )
                return
            except Exception as e:
                print(f"[SERVO] Hardware PWM initialization failed: {e}")

        # Fallback to gpiozero with appropriate pin factory
        if pwm_backend in ["gpiozero-lgpio", "pigpio", "gpiozero"]:
            try:
                if pin_factory:
                    self.servo = Servo(
                        self.servo_pin,
                        min_pulse_width=self.min_pulse_width
                        / 1000.0,  # Convert to seconds
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
            angle (float): Target angle in degrees (-45 to +45 typically)
        """
        # Clamp angle to configured range
        angle = max(self.min_angle, min(self.max_angle, angle))

        if self.servo and self.servo_initialized:
            try:
                if isinstance(self.servo, HardwarePWMServo):
                    # Use hardware PWM servo
                    self.servo.set_angle(angle, self.min_angle, self.max_angle)
                else:
                    # Use gpiozero servo
                    servo_value = self._angle_to_servo_value(angle)
                    self.servo.value = servo_value

                self.current_angle = angle
                print(f"[SERVO] Set angle: {angle:.1f}Â°")

            except Exception as e:
                print(f"[SERVO] Error setting angle: {e}")
        else:
            print(f"[SERVO] Mock mode - would set angle to {angle:.1f}Â°")
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
                if isinstance(self.servo, HardwarePWMServo):
                    self.servo.cleanup()
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
