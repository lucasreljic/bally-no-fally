"""AprilTag plate roll and pitch detector."""

import cv2
import numpy as np
from pupil_apriltags import Detector


class AprilTagPlateDetector:
    """AprilTag-based plate roll and pitch detection system."""

    def __init__(
        self,
        camera_calibration_path="camera_calibration.npz",
        tag_size=0.032,
        avg_frames=5,
    ):
        """Initialize the detector with camera parameters and configuration.

        Args:
            camera_calibration_path (str): Path to camera calibration file
            tag_size (float): Tag side length in meters
            avg_frames (int): Number of frames to average over
        """
        # Configuration constants
        self.TAG_SIZE = tag_size
        self.GROUND_TAG_IDS = [0, 1, 2]
        self.PLATFORM_TAG_IDS = [3, 4, 5]
        self.ROLL_TAG_ID = 5  # Tag whose inline direction defines the roll axis
        self.avg_frames = avg_frames

        # Load camera calibration
        self.camera_params = self._load_camera_calibration(camera_calibration_path)

        # Initialize AprilTag detector
        self.detector = Detector(
            families="tag36h11",
            nthreads=6,
            quad_decimate=1.0,
            refine_edges=1,
            decode_sharpening=0.25,
        )

        # State variables
        self.last_known_positions = {}
        self.pitch_values = []
        self.roll_values = []
        self.frame_counter = 0

    def _load_camera_calibration(self, npz_path):
        """Load camera calibration parameters from file."""
        try:
            with np.load(npz_path) as data:
                camera_matrix = data["camera_matrix"]

                fx = camera_matrix[0, 0]
                fy = camera_matrix[1, 1]
                cx = camera_matrix[0, 2]
                cy = camera_matrix[1, 2]
                camera_params = (fx, fy, cx, cy)

                print("Loaded camera intrinsics:")
                print("fx, fy, cx, cy =", camera_params)
                return camera_params
        except Exception as e:
            print(f"Error loading camera intrinsics from {npz_path}: {e}")
            exit(1)

    def compute_plane_normal(self, points):
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

    def compute_pitch_roll_from_normals(
        self, n_ground, n_plate, plate_pts, tag_positions
    ):
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

        if self.ROLL_TAG_ID in tag_positions:
            roll_dir = tag_positions[self.ROLL_TAG_ID] - centroid
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

    def detect_tags(self, frame):
        """Detect AprilTags in the given frame and return tag positions."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # More aggressive preprocessing
        gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=0)
        gray = cv2.medianBlur(gray, 5)  # Better noise reduction
        gray = cv2.bilateralFilter(gray, 2, 75, 75)  # Edge-preserving smoothing

        detections = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=self.camera_params,
            tag_size=self.TAG_SIZE,
        )

        tag_positions = {}
        detected_tag_ids = set()

        for det in detections:
            tag_id = det.tag_id
            t_cam_tag = np.array(det.pose_t).flatten()
            tag_positions[tag_id] = t_cam_tag
            detected_tag_ids.add(tag_id)
            # Update last known position for this tag
            self.last_known_positions[tag_id] = t_cam_tag

            # Draw detection on frame
            corners = det.corners.astype(int)
            cv2.polylines(frame, [corners], True, (0, 255, 0), 2)  # Green for detected
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

        return tag_positions, detected_tag_ids

    def find_and_draw_plate_center(self, frame, tag_positions):
        """Find the center of the three plate tags and draw it on the frame.

        Args:
            frame: The video frame to draw on
            tag_positions: Dictionary of detected tag positions in 3D space

        Returns:
            center_2d: The 2D pixel coordinates of the center point, or None if not all tags available
        """
        # Check if all platform tags are available (either detected or from last known positions)
        available_platform_tags = []
        for tag_id in self.PLATFORM_TAG_IDS:
            if tag_id in tag_positions:
                available_platform_tags.append(tag_id)
            elif tag_id in self.last_known_positions:
                available_platform_tags.append(tag_id)

        if len(available_platform_tags) == 3:
            # Get 3D positions of all three platform tags
            plate_positions_3d = []
            for tag_id in self.PLATFORM_TAG_IDS:
                if tag_id in tag_positions:
                    plate_positions_3d.append(tag_positions[tag_id])
                else:
                    plate_positions_3d.append(self.last_known_positions[tag_id])

            # Calculate the center point in 3D space
            center_3d = np.mean(plate_positions_3d, axis=0)

            # Project the 3D center point to 2D image coordinates
            fx, fy, cx, cy = self.camera_params

            # Simple perspective projection
            if center_3d[2] > 0:  # Make sure the point is in front of the camera
                u = fx * (center_3d[0] / center_3d[2]) + cx
                v = fy * (center_3d[1] / center_3d[2]) + cy
                center_2d = (int(u), int(v))

                # Draw the center point on the frame
                cv2.circle(frame, center_2d, 8, (255, 0, 0), -1)  # Blue filled circle
                cv2.circle(frame, center_2d, 12, (255, 255, 255), 2)  # White outer ring

                # Add text label
                cv2.putText(
                    frame,
                    "Plate Center",
                    (center_2d[0] - 40, center_2d[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

                return center_2d

        return None

    def process_frame(self, frame):
        """Process a single frame and return pitch/roll if available."""
        tag_positions, detected_tag_ids = self.detect_tags(frame)

        # Check if we have current detections or can use last known positions
        ground_found = all(t in tag_positions for t in self.GROUND_TAG_IDS)
        platform_found = all(t in tag_positions for t in self.PLATFORM_TAG_IDS)

        # If some tags are missing but we have last known positions, use them
        ground_available = all(
            t in tag_positions or t in self.last_known_positions
            for t in self.GROUND_TAG_IDS
        )
        platform_available = all(
            t in tag_positions or t in self.last_known_positions
            for t in self.PLATFORM_TAG_IDS
        )

        # Fill in missing tags with last known positions
        complete_tag_positions = tag_positions.copy()
        for tag_id in self.GROUND_TAG_IDS + self.PLATFORM_TAG_IDS:
            if (
                tag_id not in complete_tag_positions
                and tag_id in self.last_known_positions
            ):
                complete_tag_positions[tag_id] = self.last_known_positions[tag_id]

        if ground_available and platform_available:
            ground_pts = [complete_tag_positions[t] for t in self.GROUND_TAG_IDS]
            plate_pts = [complete_tag_positions[t] for t in self.PLATFORM_TAG_IDS]

            n_ground = self.compute_plane_normal(ground_pts)
            n_plate = self.compute_plane_normal(plate_pts)

            if n_ground is not None and n_plate is not None:
                # Make normals roughly point in the same direction
                if np.dot(n_ground, n_plate) < 0:
                    n_plate = -n_plate

                pitch, roll = self.compute_pitch_roll_from_normals(
                    n_ground, n_plate, plate_pts, complete_tag_positions
                )

                self.pitch_values.append(pitch)
                self.roll_values.append(roll)
                self.frame_counter += 1

                if self.frame_counter >= self.avg_frames:
                    avg_pitch = np.mean(self.pitch_values[-self.avg_frames :])
                    avg_roll = np.mean(self.roll_values[-self.avg_frames :])

                    # Indicate if using stored positions
                    using_stored = not (ground_found and platform_found)
                    status_msg = " (using stored positions)" if using_stored else ""

                    print(
                        f"Average over last {self.avg_frames} frames → Pitch: {avg_pitch:.2f}°, Roll: {avg_roll:.2f}°{status_msg}"
                    )
                    self.frame_counter = 0
                    return avg_pitch, avg_roll, using_stored, tag_positions

                return pitch, roll, not (ground_found and platform_found), tag_positions

            else:
                print(
                    "Could not compute one of the plane normals (points nearly collinear)."
                )
        else:
            missing_ground = [
                t for t in self.GROUND_TAG_IDS if t not in complete_tag_positions
            ]
            missing_platform = [
                t for t in self.PLATFORM_TAG_IDS if t not in complete_tag_positions
            ]

            if missing_ground:
                print(f"Missing ground tags (no stored positions): {missing_ground}")
            if missing_platform:
                print(
                    f"Missing platform tags (no stored positions): {missing_platform}"
                )

        return None, None, False, None

    def detect_from_frame(self, frame):
        """Return averaged pitch and roll (deg) from an external camera frame."""
        return self.process_frame(frame)

    def run(self, camera_index=2):
        """Run the main detection loop."""
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("Could not open camera.")
            return

        print("Press 'q' to quit.\n")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Camera frame not received.")
                    break

                # Process the frame
                self.process_frame(frame)

                cv2.imshow("AprilTag Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


def main():
    """Main function to run the AprilTag detector."""
    detector = AprilTagPlateDetector()
    detector.run()


if __name__ == "__main__":
    main()
