import cv2
import numpy as np
from pyray import *
from pyray import ffi
from ultralytics import YOLO
import mediapipe as mp
import pickle
from pathlib import Path
import subprocess
import sys


class ProjectorInterface:
    def __init__(self, camera_width=1280, camera_height=720):
        self.camera_width = camera_width
        self.camera_height = camera_height

        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

        # Load YOLO model (keep for potential object detection)
        print("Loading YOLO model...")
        self.model = YOLO('yolov8n.pt')
        print("Model loaded!")

        # Initialize MediaPipe Hand tracking
        print("Initializing MediaPipe Hand tracking...")
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        print("Hand tracking initialized!")

        # Detection state
        self.detections = []
        self.hand_landmarks = []  # Store hand landmarks
        self.frame_count = 0
        self.detect_every_n_frames = 1  # Run hand detection every frame
        self.confidence_threshold = 0.5

        # Projection window settings
        self.projection_width = 1920
        self.projection_height = 1080

        # Calibration state
        self.calibration_mode = False  # Start showing calibration dots
        self.corner_colors = {
            'top_left': (255, 0, 0),      # Red
            'top_right': (0, 255, 0),     # Green
            'bottom_right': (255, 0, 255), # Magenta
            'bottom_left': (255, 255, 0)  # Yellow
        }
        self.detected_corners = None  # Will store pixel locations in camera frame
        self.warped_frame = None
        self.debug_masks = {}

        # Shared frame data
        self.current_camera_frame = None

        # State files for IPC
        self.state_file = Path("/tmp/projector_state.pkl")
        self.frame_file = Path("/tmp/projector_frame.npy")
        self.dots_file = Path("/tmp/projector_dots.pkl")

        # Homography matrix for camera-to-projector mapping
        self.homography_matrix = None

    def detect_color_markers(self, frame):
        """Detect colored markers in the camera frame"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        corners = {}
        self.debug_masks = {}

        # HSV ranges for each color - more permissive ranges
        color_ranges = {
            'top_left': [(0, 50, 50), (10, 255, 255)],        # Red
            'top_right': [(40, 50, 50), (80, 255, 255)],      # Green
            'bottom_right': [(140, 50, 50), (170, 255, 255)], # Magenta
            'bottom_left': [(15, 50, 50), (35, 255, 255)]     # Yellow
        }

        for corner_name, (lower, upper) in color_ranges.items():
            lower = np.array(lower)
            upper = np.array(upper)

            mask = cv2.inRange(hsv, lower, upper)
            self.debug_masks[corner_name] = mask

            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                # Get the largest contour
                largest = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(largest)

                # Only consider if area is large enough
                if area > 100:
                    M = cv2.moments(largest)

                    if M["m00"] > 0:
                        cx = int(M["m10"] / M["m00"])
                        cy = int(M["m01"] / M["m00"])
                        corners[corner_name] = (cx, cy)

        # Only print when we actually try to calibrate
        if len(corners) > 0:
            print(f"Detected {len(corners)}/4 corners: {list(corners.keys())}")

        # Only return if all 4 corners detected
        if len(corners) == 4:
            return np.array([
                corners['top_left'],
                corners['top_right'],
                corners['bottom_right'],
                corners['bottom_left']
            ], dtype="float32")

        return None

    def compute_homography(self):
        """Compute homography matrix from camera space to projector space"""
        if self.detected_corners is None:
            return None

        # Read projector dot positions (stored as relative 0-1 coordinates)
        try:
            with open(self.dots_file, 'rb') as f:
                dot_positions_relative = pickle.load(f)

            # Convert relative to absolute pixel coordinates
            proj_points = np.array([
                [dot_positions_relative['red'][0] * self.projection_width,
                 dot_positions_relative['red'][1] * self.projection_height],
                [dot_positions_relative['green'][0] * self.projection_width,
                 dot_positions_relative['green'][1] * self.projection_height],
                [dot_positions_relative['magenta'][0] * self.projection_width,
                 dot_positions_relative['magenta'][1] * self.projection_height],
                [dot_positions_relative['yellow'][0] * self.projection_width,
                 dot_positions_relative['yellow'][1] * self.projection_height]
            ], dtype="float32")

            # Camera coordinates (where we detected the dots)
            cam_points = self.detected_corners

            print("\n=== HOMOGRAPHY CALCULATION ===")
            print(f"Camera points (pixels):\n{cam_points}")
            print(f"Projector points (pixels):\n{proj_points}")

            # Compute homography: maps camera coords -> projector coords
            self.homography_matrix, _ = cv2.findHomography(cam_points, proj_points)

            print(f"Homography matrix:\n{self.homography_matrix}")

            return self.homography_matrix

        except Exception as e:
            print(f"Error computing homography: {e}")
            import traceback
            traceback.print_exc()
            return None

    def warp_perspective(self, frame):
        """Apply perspective transform to get projector view"""
        if self.homography_matrix is None:
            return None

        try:
            # Warp entire camera frame to projector space
            warped = cv2.warpPerspective(
                frame,
                self.homography_matrix,
                (self.projection_width, self.projection_height)
            )

            # Validate the warped frame
            if warped is None or warped.size == 0:
                print("⚠ Warped frame is empty!")
                return None

            # Check if image has valid data
            if warped.min() == warped.max():
                print("⚠ Warped frame has no variance (all same color)")
                return None

            return warped

        except Exception as e:
            print(f"Error in warp_perspective: {e}")
            import traceback
            traceback.print_exc()
            return None

    def process_frame(self):
        """Capture and process a frame from the camera"""
        ret, frame = self.cap.read()
        if not ret:
            return None

        self.current_camera_frame = frame.copy()

        # Write state to file for projection window
        try:
            state = {
                'calibrated': self.calibration_mode,
                'detections': self.detections,
                'hand_landmarks': self.hand_landmarks
            }
            with open(self.state_file, 'wb') as f:
                pickle.dump(state, f)
        except Exception as e:
            print(f"Error writing state: {e}")

        # If calibrated, warp the frame and run detection
        if self.calibration_mode and self.homography_matrix is not None:
            self.warped_frame = self.warp_perspective(frame)

            # Debug: print once when starting detection
            if self.frame_count == 1:
                print(f"\n=== DETECTION ACTIVE ===")
                print(f"Calibration mode: {self.calibration_mode}")
                print(f"Homography matrix exists: {self.homography_matrix is not None}")
                print(f"Warped frame shape: {self.warped_frame.shape if self.warped_frame is not None else 'None'}")

                # Save debug image
                if self.warped_frame is not None:
                    cv2.imwrite('/tmp/warped_debug.jpg', self.warped_frame)
                    print(f"Saved debug warped frame to /tmp/warped_debug.jpg")

            # Run hand detection on warped frame
            if self.warped_frame is not None:
                # Convert BGR to RGB for MediaPipe
                warped_rgb = cv2.cvtColor(self.warped_frame, cv2.COLOR_BGR2RGB)
                results = self.hands.process(warped_rgb)

                self.hand_landmarks = []

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Convert normalized coordinates to pixel coordinates
                        landmarks_px = []
                        for landmark in hand_landmarks.landmark:
                            x_px = int(landmark.x * self.projection_width)
                            y_px = int(landmark.y * self.projection_height)
                            landmarks_px.append((x_px, y_px))

                        self.hand_landmarks.append(landmarks_px)

                # Debug output every 30 frames
                if self.frame_count % 30 == 0:
                    if len(self.hand_landmarks) > 0:
                        print(f"✓ Detected {len(self.hand_landmarks)} hand(s)")
                    else:
                        print(f"○ No hands detected (frame {self.frame_count})")

            # Save warped frame
            try:
                if self.warped_frame is not None:
                    np.save(self.frame_file, self.warped_frame)
            except Exception as e:
                print(f"Error writing frame: {e}")
        elif self.calibration_mode:
            # Calibrated but no homography matrix
            if self.frame_count % 60 == 0:
                print(f"⚠ Calibrated but homography_matrix is None!")

        self.frame_count += 1
        return frame

    def create_texture_from_frame(self, frame):
        """Helper to convert numpy frame to Raylib texture"""
        if frame is None:
            return None

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Use update_texture approach instead of load_image_from_memory
        # Create image structure directly
        img = Image(
            ffi.from_buffer("unsigned char[]", frame_rgb.tobytes()),
            frame.shape[1],
            frame.shape[0],
            1,
            PIXELFORMAT_UNCOMPRESSED_R8G8B8
        )

        texture = load_texture_from_image(img)
        return texture

    def run(self):
        """Main loop - run dev window and launch projection window separately"""
        # Initialize state file to show calibration dots
        initial_state = {
            'calibrated': False,
            'detections': [],
            'hand_landmarks': []
        }
        with open(self.state_file, 'wb') as f:
            pickle.dump(initial_state, f)

        # Launch projection window in separate process
        projection_script = Path(__file__).parent / "projection_window.py"
        subprocess.Popen([sys.executable, str(projection_script)])

        # Launch camera preview window in separate process
        camera_preview_script = Path(__file__).parent / "camera_preview.py"
        subprocess.Popen([sys.executable, str(camera_preview_script)])

        # Run dev window - disable Raylib logging
        set_trace_log_level(LOG_WARNING)  # Only show warnings and errors
        set_config_flags(FLAG_WINDOW_RESIZABLE)
        init_window(1280, 720, b"Dev Window - Camera View")
        set_target_fps(60)

        while not window_should_close():
            # Process camera frame
            self.process_frame()

            # Handle calibration
            if is_key_pressed(KEY_SPACE):
                print("\n=== SPACE KEY PRESSED ===")
                if self.current_camera_frame is not None:
                    print("Calibrating... analyzing frame")
                    corners = self.detect_color_markers(self.current_camera_frame)
                    if corners is not None:
                        self.detected_corners = corners
                        # Compute homography matrix
                        if self.compute_homography() is not None:
                            self.calibration_mode = True
                            print("✓ Calibration successful! Homography matrix computed")
                            print(f"Camera corners: {corners}")
                        else:
                            print("✗ Failed to compute homography matrix")
                    else:
                        print("✗ Calibration failed - couldn't detect all 4 colored markers")
                        print("Check if all 4 colored dots are visible in camera view")
                else:
                    print("✗ No camera frame available")

            if is_key_pressed(KEY_R):
                self.detected_corners = None
                self.calibration_mode = False
                self.homography_matrix = None
                print("Reset calibration")

            begin_drawing()
            clear_background(BLACK)

            window_width = get_screen_width()
            window_height = get_screen_height()

            # Draw camera view (full window)
            if self.current_camera_frame is not None:
                camera_texture = self.create_texture_from_frame(self.current_camera_frame)
                if camera_texture:
                    scale = min(window_width / camera_texture.width, window_height / camera_texture.height)
                    scaled_w = int(camera_texture.width * scale)
                    scaled_h = int(camera_texture.height * scale)
                    offset_x = (window_width - scaled_w) // 2
                    offset_y = (window_height - scaled_h) // 2

                    draw_texture_ex(camera_texture, Vector2(offset_x, offset_y), 0.0, scale, WHITE)

                    # Draw detected corner markers
                    if self.detected_corners is not None:
                        magenta = Color(255, 0, 255, 255)
                        colors_ray = [RED, GREEN, magenta, YELLOW]
                        for i, corner in enumerate(self.detected_corners):
                            x = int(corner[0] * scale) + offset_x
                            y = int(corner[1] * scale) + offset_y
                            draw_circle(x, y, 8, colors_ray[i])
                            draw_circle_lines(x, y, 12, colors_ray[i])

                    unload_texture(camera_texture)

            # Draw status
            status_y = 10
            if self.calibration_mode:
                draw_text(b"Status: CALIBRATED", 10, status_y, 20, GREEN)
                # Show hand count
                hand_count = f"Hands detected: {len(self.hand_landmarks)}"
                draw_text(hand_count.encode(), 10, status_y + 25, 18, YELLOW)
            else:
                draw_text(b"Status: NOT CALIBRATED", 10, status_y, 20, RED)
                draw_text(b"Press SPACE to calibrate", 10, status_y + 25, 16, LIGHTGRAY)

            # Show debug info for color detection
            if self.debug_masks:
                debug_y = 40
                magenta = Color(255, 0, 255, 255)
                colors_text = [
                    (b"Red (TL)", RED),
                    (b"Green (TR)", GREEN),
                    (b"Magenta (BR)", magenta),
                    (b"Yellow (BL)", YELLOW)
                ]
                for i, (text, color) in enumerate(colors_text):
                    draw_text(text, window_width - 150, debug_y + i * 25, 16, color)

            draw_text(b"[SPACE] Calibrate | [R] Reset | [ESC] Exit", 10, window_height - 30, 16, LIGHTGRAY)
            draw_fps(window_width - 80, 10)

            end_drawing()

        # Cleanup
        self.cap.release()
        close_window()


def main():
    interface = ProjectorInterface()
    interface.run()


if __name__ == "__main__":
    main()
