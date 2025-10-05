"""AprilTag detection and pose estimation for ball-beam balancing.

- Shows each tag's x,y,z (meters) in the camera frame
- Draws pose axes for visual debugging
- Logs pose and Euler angles to stdout
"""


import cv2
import numpy as np
from pupil_apriltags import Detector


def load_intrinsics(npz_path: str):
    """Load camera intrinsic parameters from .npz calibration file.

    Args:
        npz_path: Path to .npz file containing 'camera_matrix' and 'dist_coeffs'

    Returns:
        K:   3x3 camera matrix (float64)
        dist: (N,) distortion coefficients array (float64)
        intr: tuple (fx, fy, cx, cy)
    """
    data = np.load(npz_path)
    K = np.asarray(data["camera_matrix"], float).reshape(3, 3)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    dist = np.asarray(data["dist_coeffs"], float).ravel()
    return K, dist, (fx, fy, cx, cy)


def rmat_to_euler_zyx(R: np.ndarray):
    """Convert a rotation matrix to (yaw, pitch, roll) in radians, ZYX convention.

    Args:
        R: 3x3 rotation matrix

    Returns:
        (yaw, pitch, roll) in radians
    """
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        yaw = np.arctan2(R[1, 0], R[0, 0])
        pitch = np.arctan2(-R[2, 0], sy)
        roll = np.arctan2(R[2, 1], R[2, 2])
    else:
        # Gimbal lock
        yaw = np.arctan2(-R[0, 1], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        roll = 0.0

    return yaw, pitch, roll


def draw_pose_axes(img, R: np.ndarray, t: np.ndarray, intrinsics, axis_len=0.03):
    """Draw 3D coordinate axes on the image to visualize AprilTag pose.

    Args:
        img: BGR image
        R:   3x3 rotation matrix
        t:   (3,) translation vector in meters (camera frame)
        intrinsics: (fx, fy, cx, cy)
        axis_len: length of axis lines in meters

    Returns:
        The input image with axes drawn (X=red, Y=green, Z=blue)
    """
    fx, fy, cx, cy = intrinsics
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=float)
    dist = np.zeros(5)

    obj = np.float32(
        [[0, 0, 0], [axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]]  # X  # Y
    )  # Z

    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    tvec = t.reshape(3, 1).astype(np.float64)
    imgpts, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    imgpts = imgpts.reshape(-1, 2).astype(int)

    o, x, y, z = imgpts
    cv2.line(img, tuple(o), tuple(x), (0, 0, 255), 2)
    cv2.line(img, tuple(o), tuple(y), (0, 255, 0), 2)
    cv2.line(img, tuple(o), tuple(z), (255, 0, 0), 2)

    return img


def draw_tag_outline(img, d):
    """Draw the detected tag's quadrilateral and center."""
    corners = d.corners.astype(int)
    for i in range(4):
        p1 = tuple(corners[i])
        p2 = tuple(corners[(i + 1) % 4])
        cv2.line(img, p1, p2, (255, 255, 0), 2)
    center = tuple(map(int, d.center))
    cv2.circle(img, center, 5, (0, 0, 255), -1)
    return img


def fmt_xyz(t: np.ndarray):
    """Return x, y, z (floats) from t (3,)."""
    x, y, z = float(t[0]), float(t[1]), float(t[2])
    return x, y, z


def detect_apriltag_poses(frame, detector, intrinsics, tag_size_m=0.045):
    """Detect AprilTags in a frame and return their pose data.

    Args:
        frame: BGR image from camera
        detector: pupil_apriltags Detector instance
        intrinsics: (fx, fy, cx, cy) camera parameters
        tag_size_m: black-square edge length of the tag (meters)

    Returns:
        List of dictionaries, each containing:
        {
            'tag_id': int,
            'x': float,         # meters
            'y': float,         # meters
            'z': float,         # meters
            'distance': float,  # meters
            'yaw': float,       # degrees
            'pitch': float,     # degrees
            'roll': float,      # degrees
            'pose_R': np.ndarray,  # 3x3 rotation matrix
            'pose_t': np.ndarray,  # 3x1 translation vector
            'center': tuple,    # (x, y) pixel coordinates of tag center
            'corners': np.ndarray  # 4x2 corner coordinates
        }
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fx, fy, cx, cy = intrinsics

    # Detect + estimate pose
    detections = detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=(fx, fy, cx, cy),
        tag_size=tag_size_m,
    )

    pose_data = []
    for d in detections:
        # Pose in camera frame
        R = np.asarray(d.pose_R, float)
        t = np.asarray(d.pose_t, float).reshape(3)
        x, y, z = fmt_xyz(t)
        dist_m = float(np.linalg.norm(t))
        yaw, pitch, roll = np.degrees(rmat_to_euler_zyx(R))

        pose_info = {
            "tag_id": d.tag_id,
            "x": x,
            "y": y,
            "z": z,
            "distance": dist_m,
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll,
            "pose_R": R,
            "pose_t": t,
            "center": tuple(map(int, d.center)),
            "corners": d.corners,
        }
        pose_data.append(pose_info)

    return pose_data


def run_pose_demo(
    npz_path="camera_calibration.npz",
    tag_size_m=0.045,
    cam_index=0,
    families="tag36h11",
    decimate=1.0,
    blur=0.0,
):
    """Real-time AprilTag pose estimation with x,y,z overlay.

    Args:
        npz_path: path to .npz camera calibration
        tag_size_m: black-square edge length of the tag (meters)
        cam_index: OpenCV camera index
        families: AprilTag family string for the Detector
        decimate: image decimation factor for detector (speed/accuracy tradeoff)
        blur:     Gaussian blur sigma for detector (0 = off)
    """
    # Load intrinsics
    K, dist, (fx, fy, cx, cy) = load_intrinsics(npz_path)
    intr = (fx, fy, cx, cy)

    # Create detector
    det = Detector(
        families=families,
        nthreads=0,
        quad_decimate=decimate,
        quad_sigma=blur,
        refine_edges=True,
    )

    # Open camera
    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera")

    print("Press 'q' or ESC to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            pose_data = detect_apriltag_poses(frame, det, intr, tag_size_m)

            for pose_info in pose_data:
                tag_id = pose_info["tag_id"]
                x, y, z = pose_info["x"], pose_info["y"], pose_info["z"]
                dist_m = pose_info["distance"]
                yaw, pitch, roll = (
                    pose_info["yaw"],
                    pose_info["pitch"],
                    pose_info["roll"],
                )
                R, t = pose_info["pose_R"], pose_info["pose_t"]
                center = pose_info["center"]
                corners = pose_info["corners"]

                class DetectionForDrawing:
                    def __init__(self, tag_id, center, corners):
                        self.tag_id = tag_id
                        self.center = center
                        self.corners = corners

                d_draw = DetectionForDrawing(tag_id, center, corners)

                draw_tag_outline(frame, d_draw)
                cv2.putText(
                    frame,
                    f"id:{tag_id}",
                    (center[0] + 6, center[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                draw_pose_axes(frame, R, t, intr, axis_len=tag_size_m * 0.5)

                cv2.putText(
                    frame,
                    f"x={x:+.3f} m  y={y:+.3f} m  z={z:+.3f} m",
                    (center[0] + 6, center[1] + 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (240, 240, 240),
                    1,
                )
                cv2.putText(
                    frame,
                    f"dist={dist_m:.3f} m  yaw={yaw:+.1f}  pitch={pitch:+.1f}  roll={roll:+.1f}",
                    (center[0] + 6, center[1] + 36),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                )

                print(
                    f"tag_id={tag_id}  "
                    f"x={x:.6f}  y={y:.6f}  z={z:.6f}  dist={dist_m:.6f}  "
                    f"yaw={yaw:.3f}  pitch={pitch:.3f}  roll={roll:.3f}"
                )

            cv2.imshow("AprilTag Pose (x,y,z in meters)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    cal_path = "camera_calibration.npz"
    print(f"Using calibration file: {cal_path}")
    run_pose_demo(npz_path=cal_path, tag_size_m=0.045, cam_index=0)
