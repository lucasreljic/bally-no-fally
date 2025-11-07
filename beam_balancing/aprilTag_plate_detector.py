"""AprilTag plate roll and pitch detector."""

import cv2
import numpy as np
from pupil_apriltags import Detector

TAG_SIZE = 0.032  # Tag side length in meters
npz_path = "camera_calibration.npz"

GROUND_TAG_IDS = [0, 1, 2]
PLATFORM_TAG_IDS = [3, 4, 5]
ROLL_TAG_ID = 5  # Tag whose inline direction defines the roll axis on the platform

# Number of frames to average over
avg_frames = 5

try:
    with np.load(npz_path) as data:
        camera_matrix = data["camera_matrix"]
        dist_coeffs = data["dist_coeffs"]

        fx = camera_matrix[0, 0]
        fy = camera_matrix[1, 1]
        cx = camera_matrix[0, 2]
        cy = camera_matrix[1, 2]
        CAMERA_PARAMS = (fx, fy, cx, cy)

        print("Loaded camera intrinsics:")
        print("fx, fy, cx, cy =", CAMERA_PARAMS)
except Exception as e:
    print(f"Error loading camera intrinsics from {npz_path}: {e}")
    exit(1)


def compute_plane_normal(points):
    """Compute the unit normal vector of a plane defined by three points.

    Uses the cross product of two vectors in the plane to find the normal vector,
    which is then normalized to unit length.

    Args:
        points (list): List of three 3D points (numpy arrays) defining the plane.
                       Points should be in the format [p0, p1, p2] where each
                       point is a numpy array of shape (3,).

    Returns:
        numpy.ndarray or None: Unit normal vector of the plane as a numpy array
                               of shape (3,), or None if the points are nearly
                               collinear (cannot define a unique plane).
    """
    p0, p1, p2 = points
    v1 = p1 - p0
    v2 = p2 - p0
    n_raw = np.cross(v1, v2)
    norm_n = np.linalg.norm(n_raw)
    if norm_n < 1e-9:
        return None
    return n_raw / norm_n  # unit normal


def compute_pitch_roll_from_normals(n_ground, n_plate, plate_pts, tag_positions):
    """Calculate pitch and roll angles of the platform relative to the ground.

    - 'Pitch' corresponds to rotation about the axis aligned with tag ID 5 (inline axis)
    - 'Roll' corresponds to rotation about the perpendicular axis in the platform plane
    """
    # ---- Ground frame basis ----
    z_g = n_ground
    x0 = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(x0, z_g)) > 0.9:
        x0 = np.array([0.0, 1.0, 0.0])

    x_g = x0 - np.dot(x0, z_g) * z_g
    x_g = x_g / np.linalg.norm(x_g)
    y_g = np.cross(z_g, x_g)
    y_g = y_g / np.linalg.norm(y_g)

    R_cam_g = np.column_stack((x_g, y_g, z_g))

    # ---- Platform frame basis ----
    z_p = n_plate
    plate_pts_arr = np.vstack(plate_pts)
    centroid = plate_pts_arr.mean(axis=0)

    if 5 in tag_positions:
        roll_dir = tag_positions[5] - centroid
    else:
        roll_dir = plate_pts_arr[0] - centroid

    roll_dir = roll_dir - np.dot(roll_dir, z_p) * z_p
    roll_dir /= np.linalg.norm(roll_dir)
    x_p = roll_dir
    y_p = np.cross(z_p, x_p)
    y_p = y_p / np.linalg.norm(y_p)

    R_cam_p = np.column_stack((x_p, y_p, z_p))
    R_g_p = R_cam_g.T @ R_cam_p

    # Extract (flipped) pitch and roll
    r20 = np.clip(R_g_p[2, 0], -1.0, 1.0)
    roll_rad = -np.arcsin(r20)  # swapped
    pitch_rad = np.arctan2(R_g_p[2, 1], R_g_p[2, 2])  # swapped

    roll_deg = np.degrees(roll_rad)
    roll_deg = -roll_deg
    pitch_deg = np.degrees(pitch_rad)

    return pitch_deg, roll_deg


# -------------------------------
# AprilTag Detector setup
# -------------------------------
detector = Detector(
    families="tag36h11",
    nthreads=4,
    quad_decimate=1.0,
    refine_edges=1,
    decode_sharpening=0.25,
)

# -------------------------------
# Video capture
# -------------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Could not open camera.")
    exit(1)

print("Press 'q' to quit.\n")

pitch_values = []
roll_values = []
frame_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera frame not received.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    detections = detector.detect(
        gray, estimate_tag_pose=True, camera_params=CAMERA_PARAMS, tag_size=TAG_SIZE
    )

    tag_positions = {}
    for det in detections:
        tag_id = det.tag_id
        t_cam_tag = np.array(det.pose_t).flatten()
        tag_positions[tag_id] = t_cam_tag

        corners = det.corners.astype(int)
        cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
        cX, cY = corners.mean(axis=0).astype(int)
        cv2.putText(
            frame,
            f"ID:{tag_id}",
            (cX - 15, cY - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    ground_found = all(t in tag_positions for t in GROUND_TAG_IDS)
    platform_found = all(t in tag_positions for t in PLATFORM_TAG_IDS)

    if ground_found and platform_found:
        ground_pts = [tag_positions[t] for t in GROUND_TAG_IDS]
        plate_pts = [tag_positions[t] for t in PLATFORM_TAG_IDS]

        n_ground = compute_plane_normal(ground_pts)
        n_plate = compute_plane_normal(plate_pts)

        if n_ground is not None and n_plate is not None:
            # Make normals roughly point in the same direction
            if np.dot(n_ground, n_plate) < 0:
                n_plate = -n_plate

            pitch, roll = compute_pitch_roll_from_normals(
                n_ground, n_plate, plate_pts, tag_positions
            )

            pitch_values.append(pitch)
            roll_values.append(roll)
            frame_counter += 1

            if frame_counter >= avg_frames:
                avg_pitch = np.mean(pitch_values[-avg_frames:])
                avg_roll = np.mean(roll_values[-avg_frames:])
                print(
                    f"Average over last {avg_frames} frames → Pitch: {avg_pitch:.2f}°, Roll: {avg_roll:.2f}°"
                )
                frame_counter = 0

        else:
            print(
                "Could not compute one of the plane normals (points nearly collinear)."
            )
    else:
        if not ground_found:
            print("Missing one or more ground tags.")
        if not platform_found:
            print("Missing one or more platform tags.")

    cv2.imshow("AprilTag Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
