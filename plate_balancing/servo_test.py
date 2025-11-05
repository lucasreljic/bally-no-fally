"""Phased sine wave servo test for Stewart platform."""
import argparse
import math
import time

from servo_control import ServoController


def main(freq=0.25, amp=0.8, duration=None, rate=50.0):
    """Phased sine wave test for three servos controlling a Stewart platform.

    freq: frequency of sine wave in Hz
    amp: amplitude of sine wave (normalized -1.0 to 1.0)
    duration: total time to run the test in seconds (None for infinite)
    rate: update rate in Hz
    """
    servo_0 = ServoController(config_file="plate_balancing/servo_0.json")
    servo_1 = ServoController(config_file="plate_balancing/servo_1.json")
    servo_2 = ServoController(config_file="plate_balancing/servo_2.json")

    # center everything first
    for s in (servo_0, servo_1, servo_2):
        s.set_position_normalized(0.0)
    time.sleep(0.5)

    phases = [0.0, 2 * math.pi / 3, 4 * math.pi / 3]  # 0°, 120°, 240°
    t0 = time.time()
    dt = 1.0 / max(1.0, rate)
    try:
        while True:
            now = time.time() - t0
            if duration and now >= duration:
                break
            pos0 = amp * math.sin(2 * math.pi * freq * now + phases[0])
            pos1 = amp * math.sin(2 * math.pi * freq * now + phases[1])
            pos2 = amp * math.sin(2 * math.pi * freq * now + phases[2])
            servo_0.set_position_normalized(pos0)
            servo_1.set_position_normalized(pos1)
            servo_2.set_position_normalized(pos2)
            time.sleep(dt)
    except KeyboardInterrupt:
        pass
    finally:
        # gently center servos on exit
        for s in (servo_0, servo_1, servo_2):
            s.set_position_normalized(0.0)
        time.sleep(0.3)


if __name__ == "__main__":
    """
    Entry point for phased sine servo test.
    """
    parser = argparse.ArgumentParser(
        description="Phased sine servo test for Stewart platform"
    )
    parser.add_argument("--freq", type=float, default=0.25, help="sine frequency (Hz)")
    parser.add_argument(
        "--amp", type=float, default=0.8, help="normalized amplitude (0..1)"
    )
    parser.add_argument(
        "--duration", type=float, default=None, help="seconds to run (ctrl-c to quit)"
    )
    parser.add_argument("--rate", type=float, default=50.0, help="update rate (Hz)")
    args = parser.parse_args()
    main(freq=args.freq, amp=args.amp, duration=args.duration, rate=args.rate)
