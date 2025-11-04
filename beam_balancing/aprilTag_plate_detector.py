"""AprilTag-based plate orientation estimation.

- Detects individual AprilTags on a plate
- Calculates average roll and pitch from individual tag orientations
- Uses single reference tag (ID 3) on floor for camera-independent measurements
- Provides callbacks for angles and optional visualization frame
"""

import cv2
import numpy as np
from pupil_apriltags import Detector


class AprilTagPlateDetector:
    """Detect AprilTags on a plate and estimate its orientation."""

    def __init__(
        self,
        npz_path="camera_calibration.npz",
        tag_size_m=0.018,
        cam_index=0,
        families="tag36h11",
        decimate=1.0,
        blur=0.0,
        expected_tag_ids=None,
        reference_tag_id=3,
        angle_callback=None,
        frame_callback=None,
    ):
        """
        Args:
            npz_path: path to .npz camera calibration file
                      (must contain 'camera_matrix' and 'dist_coeffs')
            tag_size_m: black-square edge length of the tag (meters)
            cam_index: index of the camera for cv2.VideoCapture
            families: AprilTag family string for the Detector
            decimate: image decimation factor for detector
            blur: Gaussian blur sigma for detector (0 = off)
            expected_tag_ids: optional iterable of tag IDs to use (e.g. [0,1,2])
            reference_tag_id: AprilTag ID on floor for reference (e.g. 3)
            angle_callback: function(roll_deg, pitch_deg, z_m)
            frame_callback: optional function(frame) -> bool
                            if returns False, run() loop stops
        """
        self.tag_size_m = tag_size_m
        self.K, self.dist, self.intrinsics = self._load_intrinsics(npz_path)

        self.detector = Detector(
            families=families,
            nthreads=1,  # Use 1 thread for better performance than 0 (auto)
            quad_decimate=decimate,
            quad_sigma=blur,
            refine_edges=True,
            decode_sharpening=0.25,  # Reduce processing for speed
        )

        self.cap = cv2.VideoCapture(cam_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera error: could not open camera at index {cam_index}")
        
        # Optimize camera settings for performance
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer to minimize lag
        self.cap.set(cv2.CAP_PROP_FPS, 30)  # Set reasonable frame rate
        
        # Test reading a frame to ensure camera is working
        test_ok, test_frame = self.cap.read()
        if not test_ok:
            self.cap.release()
            raise RuntimeError(f"Camera error: camera {cam_index} opened but cannot read frames. Try a different camera index.")
        
        print(f"Successfully opened camera {cam_index} with resolution: {test_frame.shape[:2]}")

        self.expected_tag_ids = expected_tag_ids
        self.reference_tag_id = reference_tag_id
        self.reference_normal = None  # Will store floor plane normal
        self.angle_callback = angle_callback
        self.frame_callback = frame_callback

    def run(self):
        """
        Main loop: grab frames, estimate plate orientation, call callbacks.

        angle_callback(roll_deg, pitch_deg, z_m)
        frame_callback(vis_frame) -> bool (return False to stop loop)
        """
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    raise RuntimeError("Camera error: failed to grab frame")

                roll_deg, pitch_deg, z_m, vis_frame = self.process_frame(frame)

                # Only call angle callback when we have a valid estimate
                if self.angle_callback is not None and roll_deg is not None:
                    self.angle_callback(roll_deg, pitch_deg, z_m)

                # Always call frame callback (for live view + debugging)
                if self.frame_callback is not None:
                    if not self.frame_callback(vis_frame):
                        break
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

    def process_frame(self, frame):
        """
        Process a single frame: detect tags, estimate plate plane.

        Returns:
            roll_deg, pitch_deg, z_m, vis_frame

        Notes:
            - If fewer than 2 usable tags or geometry is degenerate:
              roll_deg, pitch_deg, z_m are None, and an error message is
              drawn on vis_frame for debugging.
        """
        tags = self._detect_tags(frame)
        
        # Separate reference tag from plate tags
        reference_tag = None
        plate_tags = []
        
        for tag in tags:
            if tag["id"] == self.reference_tag_id:
                reference_tag = tag
            else:
                plate_tags.append(tag)
        
        # Update reference plane if reference tag is detected
        if reference_tag is not None:
            self._update_reference_plane(reference_tag)

        roll_deg = pitch_deg = z_m = None
        centroid = normal = None
        error_msg = None

        if len(plate_tags) < 1:
            error_msg = "No plate tags detected."
        else:
            # Calculate average roll/pitch from individual plate tags
            try:
                roll_values = []
                pitch_values = []
                z_values = []
                
                for tag in plate_tags:
                    tag_roll, tag_pitch = self._get_tag_angles(tag)
                    if tag_roll is not None and tag_pitch is not None:
                        # Convert to reference frame if we have a reference
                        if self.reference_normal is not None:
                            tag_roll, tag_pitch = self._angles_relative_to_reference_individual(tag_roll, tag_pitch)
                        
                        roll_values.append(tag_roll)
                        pitch_values.append(tag_pitch)
                        z_values.append(tag["t"][2])
                
                if len(roll_values) > 0:
                    roll_deg = float(np.mean(roll_values))
                    pitch_deg = float(np.mean(pitch_values))
                    z_m = float(np.mean(z_values))
                else:
                    error_msg = "Could not calculate angles from any plate tags."
                    
            except ValueError as e:
                error_msg = str(e)

        vis = frame.copy()
        vis = self._draw_tags(vis, tags)
        if centroid is not None and normal is not None:
            vis = self._draw_plane_normal(vis, centroid, normal)
        
        # Draw reference tag differently if present
        if reference_tag is not None:
            vis = self._draw_reference_tag(vis, reference_tag)

        if error_msg is not None:
            cv2.putText(
                vis,
                error_msg,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )

        return roll_deg, pitch_deg, z_m, vis

    def _load_intrinsics(self, npz_path: str):
        data = np.load(npz_path)
        K = np.asarray(data["camera_matrix"], float).reshape(3, 3)
        dist = np.asarray(data["dist_coeffs"], float).ravel()
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        return K, dist, (fx, fy, cx, cy)

    def _detect_tags(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fx, fy, cx, cy = self.intrinsics

        detections = self.detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=(fx, fy, cx, cy),
            tag_size=self.tag_size_m,
        )

        # Store full detections for reuse (avoiding redundant detection calls)
        self._cached_detections = {d.tag_id: d for d in detections}

        tags = []
        for d in detections:
            if self.expected_tag_ids is not None and d.tag_id not in self.expected_tag_ids:
                continue
            t = np.asarray(d.pose_t, float).reshape(3)
            tags.append(
                {
                    "id": d.tag_id,
                    "t": t,
                    "center": tuple(map(int, d.center)),
                    "corners": d.corners.astype(int),
                }
            )
        # Sort by ID for consistency
        tags.sort(key=lambda x: x["id"])
        return tags

    def _fit_plane(self, pts):
        """
        Least-squares plane through N>=3 points (in camera frame).
        Returns centroid and unit normal (normal.z forced >= 0 to define 'forward').
        """
        P = np.stack(pts, axis=0)
        centroid = P.mean(axis=0)
        _, _, Vt = np.linalg.svd(P - centroid)
        normal = Vt[-1]
        normal /= np.linalg.norm(normal)
        if normal[2] < 0: 
            normal = -normal
        return centroid, normal

    def _plane_from_two(self, p1, p2):
        """
        Approximate plane from origin and two tag positions (2 tags case).
        """
        v1 = p1
        v2 = p2
        n = np.cross(v1, v2)
        if np.linalg.norm(n) < 1e-8:
            raise ValueError("Messed up geometry with 2 tags.")
        n /= np.linalg.norm(n)
        if n[2] < 0:
            n = -n
        centroid = 0.5 * (p1 + p2)
        return centroid, n

    def _angles_from_normal(self, n):
        """
        Compute roll, pitch (in degrees) from plane normal in camera frame.

        Camera frame (OpenCV): x right, y down, z forward.
        For a level plate facing the camera, normal ~ (0, 0, 1) -> roll=0, pitch=0.
        """
        nx, ny, nz = n
        roll = np.arctan2(ny, nz)  # rotation about x-axis
        pitch = np.arctan2(-nx, np.sqrt(ny * ny + nz * nz))  # rotation about y-axis
        return float(np.degrees(roll)), float(np.degrees(pitch))

    def _get_tag_angles(self, tag):
        """
        Get roll and pitch angles from individual tag's orientation.
        Uses cached detection results to avoid redundant detection calls.
        """
        tag_id = tag["id"]
        
        # Use cached detection result instead of re-detecting
        if hasattr(self, '_cached_detections') and tag_id in self._cached_detections:
            d = self._cached_detections[tag_id]
            # Get the rotation matrix and extract the normal (Z-axis of the tag)
            R = np.asarray(d.pose_R, dtype=float).reshape(3, 3)
            # The third column of R is the tag's Z-axis (normal direction)
            tag_normal = R[:, 2]
            
            # Calculate roll and pitch from the tag's normal
            return self._angles_from_normal(tag_normal)
        
        return None, None

    def _update_reference_plane(self, reference_tag):
        """
        Update the reference plane normal from single floor AprilTag.
        The reference tag should be placed flat on the floor/level surface.
        Uses cached detection results to avoid redundant detection calls.
        """
        tag_id = self.reference_tag_id
        
        # Use cached detection result instead of re-detecting
        if hasattr(self, '_cached_detections') and tag_id in self._cached_detections:
            d = self._cached_detections[tag_id]
            # Get the rotation matrix and extract the normal (Z-axis of the tag)
            R = np.asarray(d.pose_R, dtype=float).reshape(3, 3)
            # The third column of R is the tag's Z-axis (normal direction)
            tag_normal = R[:, 2]

                
            self.reference_normal = tag_normal / np.linalg.norm(tag_normal)

    def _angles_relative_to_reference_individual(self, tag_roll, tag_pitch):
        """
        Calculate roll and pitch relative to the reference plane for individual tag.
        """
        if self.reference_normal is None:
            return tag_roll, tag_pitch
            
        # Calculate roll and pitch of reference plane
        ref_roll, ref_pitch = self._angles_from_normal(self.reference_normal)
        
        # Relative angles = tag angles - reference angles
        relative_roll = tag_roll - ref_roll
        relative_pitch = tag_pitch - ref_pitch
        print(ref_roll, ref_pitch)
        
        return relative_roll, relative_pitch
            
    def _angles_relative_to_reference(self, plate_normal):
        """
        Calculate roll and pitch relative to the reference plane.
        """
        if self.reference_normal is None:
            return self._angles_from_normal(plate_normal)
            
        # Calculate the relative normal by comparing plate normal to reference normal
        # This gives us the orientation relative to the floor plane
        
        # Method 1: Direct angle calculation between normals
        # First, project both normals onto a common reference frame
        
        # Calculate roll and pitch of reference plane
        ref_roll, ref_pitch = self._angles_from_normal(self.reference_normal)
        
        # Calculate roll and pitch of plate
        plate_roll, plate_pitch = self._angles_from_normal(plate_normal)
        
        # Relative angles = plate angles - reference angles
        relative_roll = plate_roll - ref_roll
        relative_pitch = plate_pitch - ref_pitch
        
        return relative_roll, relative_pitch

    # ---------- visualization ----------

    def _draw_reference_tag(self, img, reference_tag):
        """Draw reference tag with different color to distinguish it."""
        corners = reference_tag["corners"]
        for i in range(4):
            p1 = tuple(corners[i])
            p2 = tuple(corners[(i + 1) % 4])
            cv2.line(img, p1, p2, (0, 255, 0), 3)  # Green for reference
        cv2.circle(img, reference_tag["center"], 6, (0, 255, 0), -1)
        cv2.putText(
            img,
            f"REF:{reference_tag['id']}",
            (reference_tag["center"][0] + 5, reference_tag["center"][1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        return img

    def _draw_tags(self, img, tags):
        """Draw tag outlines and centers."""
        for t in tags:
            corners = t["corners"]
            for i in range(4):
                p1 = tuple(corners[i])
                p2 = tuple(corners[(i + 1) % 4])
                cv2.line(img, p1, p2, (0, 255, 255), 2)
            cv2.circle(img, t["center"], 4, (0, 0, 255), -1)
            cv2.putText(
                img,
                f"id:{t['id']}",
                (t["center"][0] + 5, t["center"][1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
        return img

    def _draw_plane_normal(self, img, centroid, normal, length_scale=0.08):
        """
        Draw the plane normal as a 'forward' arrow originating from the centroid.
        """
        fx, fy, cx, cy = self.intrinsics

        def project(point3d):
            x, y, z = point3d
            if z <= 0:
                return None
            u = fx * x / z + cx
            v = fy * y / z + cy
            return int(u), int(v)

        p0 = project(centroid)
        p1 = project(centroid + normal * length_scale)

        if p0 is not None and p1 is not None:
            cv2.arrowedLine(img, p0, p1, (0, 0, 255), 2, tipLength=0.2)
            cv2.putText(
                img,
                "forward",
                (p1[0] + 5, p1[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

        return img


def find_available_cameras(max_cameras=5):
    """
    Find available camera indices.
    Returns list of working camera indices.
    """
    available_cameras = []
    print("Scanning for available cameras...")
    
    for i in range(max_cameras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print(f"Camera {i}: Available ({frame.shape[:2]})")
                available_cameras.append(i)
            else:
                print(f"Camera {i}: Opens but cannot read frames")
            cap.release()
        else:
            print(f"Camera {i}: Not available")
    
    return available_cameras


if __name__ == "__main__":
    
    # Find available cameras first
    available_cameras = find_available_cameras()
    if not available_cameras:
        print("Error: No working cameras found!")
        exit(1)
    
    print(f"\nUsing camera {available_cameras[0]}")
    
    def angle_cb(roll, pitch, z):
        print(f"roll={roll:+6.2f} deg  pitch={pitch:+6.2f} deg  z={z:.3f} m")

    def frame_cb(frame):
        cv2.imshow("AprilTag Plate", frame)
        key = cv2.waitKey(1) & 0xFF
        return key not in (27, ord("q"))  # ESC or 'q' to quit

    detector = AprilTagPlateDetector(
        npz_path="camera_calibration.npz",
        tag_size_m=0.018,
        cam_index=available_cameras[0],  # Use first available camera
        expected_tag_ids=None, 
        reference_tag_id=3,  # Place tag ID 3 on the floor as reference
        angle_callback=angle_cb,
        frame_callback=frame_cb,
    )
    detector.run()
