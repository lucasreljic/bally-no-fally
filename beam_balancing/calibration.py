"""Camera calibration module using checkerboard pattern.

This module provides functionality to capture images from a webcam and perform
camera intrinsics calibration using a checkerboard pattern.
"""

import os
from typing import List, Tuple

import cv2
import numpy as np

CHECKERBOARD_SIZE = (8, 6)  # Number of inner corners per a chessboard row and column


def capture_calibration_images(
    checkerboard_size: Tuple[int, int] = CHECKERBOARD_SIZE,
    num_images: int = 20,
    output_dir: str = "calibration_images",
) -> List[str]:
    """Capture images for camera calibration using checkerboard pattern.

    Args:
        checkerboard_size: Inner corners of the checkerboard (width, height)
        num_images: Number of images to capture for calibration
        output_dir: Directory to save captured images

    Returns:
        List of captured image file paths
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    captured_images = []
    count = 0

    print(f"Capturing {num_images} images for calibration...")
    print(
        "Press SPACE to capture image, ESC to exit early, 'c' to continue to calibration"
    )

    while count < num_images:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Find checkerboard corners
        found, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        # Draw corners if found
        if found:
            cv2.drawChessboardCorners(frame, checkerboard_size, corners, found)
            cv2.putText(
                frame,
                "Checkerboard detected! Press SPACE to capture",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                frame,
                "Move checkerboard into view",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        cv2.putText(
            frame,
            f"Captured: {count}/{num_images}",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

        cv2.imshow("Camera Calibration - Capture Images", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC key
            break
        elif key == ord(" ") and found:  # SPACE key and checkerboard found
            filename = os.path.join(output_dir, f"calibration_{count:02d}.jpg")
            cv2.imwrite(filename, frame)
            captured_images.append(filename)
            count += 1
            print(f"Captured image {count}/{num_images}: {filename}")
        elif key == ord("c") and count > 5:  # Continue with current images
            break

    cap.release()
    cv2.destroyAllWindows()

    print(f"Captured {len(captured_images)} images for calibration")
    return captured_images


def calibrate_camera(
    image_paths: List[str],
    checkerboard_size: Tuple[int, int] = CHECKERBOARD_SIZE,
    square_size: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    """Perform camera calibration using captured images.

    Args:
        image_paths: List of paths to calibration images
        checkerboard_size: Inner corners of the checkerboard (width, height)
        square_size: Size of checkerboard squares (in any unit, e.g., mm)

    Returns:
        Tuple containing (camera_matrix, distortion_coefficients, rvecs, tvecs)
    """
    # Prepare object points (3D points in real world space)
    objp = np.zeros((checkerboard_size[0] * checkerboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[
        0 : checkerboard_size[0], 0 : checkerboard_size[1]
    ].T.reshape(-1, 2)
    objp *= square_size

    # Arrays to store object points and image points from all images
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    print("Processing captured images...")

    for i, image_path in enumerate(image_paths):
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not load image: {image_path}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find checkerboard corners
        found, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

        if found:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            objpoints.append(objp)
            imgpoints.append(corners2)
            print(f"Processed image {i+1}/{len(image_paths)}")
        else:
            print(f"Could not find checkerboard in image {i+1}/{len(image_paths)}")

    if len(objpoints) == 0:
        raise ValueError("No valid checkerboard patterns found in any images")

    print(f"Calibrating camera using {len(objpoints)} valid images...")

    # Perform camera calibration
    img_shape = cv2.imread(image_paths[0]).shape[:2][::-1]  # (width, height)
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img_shape, None, None
    )

    if ret:
        print("Camera calibration successful!")
        print(f"Reprojection error: {ret:.4f}")
    else:
        print("Camera calibration failed!")

    return camera_matrix, dist_coeffs, rvecs, tvecs


def save_calibration_results(
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    filename: str = "camera_calibration.npz",
) -> None:
    """Save calibration results to file.

    Args:
        camera_matrix: Camera intrinsic matrix
        dist_coeffs: Distortion coefficients
        filename: Output filename for calibration data
    """
    np.savez(filename, camera_matrix=camera_matrix, dist_coeffs=dist_coeffs)
    print(f"Calibration results saved to: {filename}")


def load_calibration_results(
    filename: str = "camera_calibration.npz",
) -> Tuple[np.ndarray, np.ndarray]:
    """Load calibration results from file.

    Args:
        filename: Input filename for calibration data

    Returns:
        Tuple containing (camera_matrix, distortion_coefficients)
    """
    data = np.load(filename)
    return data["camera_matrix"], data["dist_coeffs"]


def main():
    """Main function to perform complete camera calibration workflow."""
    try:
        # Step 1: Capture calibration images
        print("=== Camera Calibration Script ===")
        print(
            "Make sure you have a printed checkerboard pattern ({} inner corners)".format(
                CHECKERBOARD_SIZE
            )
        )
        print("Hold the checkerboard at different angles and distances from the camera")
        input("Press Enter to start capturing images...")

        images = capture_calibration_images(
            checkerboard_size=CHECKERBOARD_SIZE, num_images=20
        )

        if len(images) < 5:
            print("Need at least 5 good images for calibration. Please run again.")
            return

        # Step 2: Perform calibration
        camera_matrix, dist_coeffs, rvecs, tvecs = calibrate_camera(images)

        # Step 3: Save results
        save_calibration_results(camera_matrix, dist_coeffs)

        # Step 4: Display results
        print("\n=== Calibration Results ===")
        print(f"Camera Matrix:\n{camera_matrix}")
        print(f"\nDistortion Coefficients:\n{dist_coeffs.ravel()}")

        # Calculate focal lengths and principal point
        fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

        print(f"\nFocal lengths: fx={fx:.2f}, fy={fy:.2f}")
        print(f"Principal point: cx={cx:.2f}, cy={cy:.2f}")

    except Exception as e:
        print(f"Error during calibration: {e}")


if __name__ == "__main__":
    main()
