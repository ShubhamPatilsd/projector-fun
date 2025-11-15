"""Projection window - runs separately"""
import pickle
import time
from pathlib import Path
from pyray import *
from pyray import ffi
import cv2
import numpy as np

# MediaPipe hand connections (landmark pairs to draw lines between)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),  # Index finger
    (0, 9), (9, 10), (10, 11), (11, 12),  # Middle finger
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (0, 17), (17, 18), (18, 19), (19, 20),  # Pinky
    (5, 9), (9, 13), (13, 17)  # Palm connections
]


def create_texture_from_frame(frame):
    """Helper to convert numpy frame to Raylib texture"""
    if frame is None:
        return None

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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


def main():
    WIDTH = 1920
    HEIGHT = 1080

    # Disable Raylib verbose logging
    set_trace_log_level(LOG_WARNING)

    # Allow fullscreen and resizing
    set_config_flags(FLAG_WINDOW_RESIZABLE)
    init_window(WIDTH, HEIGHT, b"Projection Window")
    set_target_fps(60)

    state_file = Path("/tmp/projector_state.pkl")
    frame_file = Path("/tmp/projector_frame.npy")
    dots_file = Path("/tmp/projector_dots.pkl")

    # Default dot positions as RELATIVE coordinates (0-1 range)
    # All in top-right corner
    default_dots_relative = {
        'red': [0.85, 0.10],      # top-left
        'green': [0.95, 0.10],    # top-right
        'magenta': [0.95, 0.20],  # bottom-right
        'yellow': [0.85, 0.20]    # bottom-left
    }

    # Load saved relative positions or use defaults
    if dots_file.exists():
        try:
            with open(dots_file, 'rb') as f:
                dot_positions_relative = pickle.load(f)
        except:
            dot_positions_relative = default_dots_relative
    else:
        dot_positions_relative = default_dots_relative

    # Convert relative to absolute positions
    def get_absolute_positions(width, height):
        return {
            name: [int(pos[0] * width), int(pos[1] * height)]
            for name, pos in dot_positions_relative.items()
        }

    dot_positions = get_absolute_positions(WIDTH, HEIGHT)

    # Dragging state
    dragging_dot = None
    dot_radius = 60

    while not window_should_close():
        # Toggle fullscreen with F11
        if is_key_pressed(KEY_F11):
            toggle_fullscreen()

        # Get actual window dimensions
        window_width = get_screen_width()
        window_height = get_screen_height()

        # Calculate scale factors for rendering
        scale_x = window_width / WIDTH
        scale_y = window_height / HEIGHT

        # Update absolute positions based on actual window size
        dot_positions = get_absolute_positions(window_width, window_height)

        # Handle mouse dragging for calibration dots
        mouse_pos = get_mouse_position()
        mouse_x_window = mouse_pos.x
        mouse_y_window = mouse_pos.y

        # Convert mouse position to 1920x1080 space
        mouse_x = mouse_x_window / window_width * WIDTH
        mouse_y = mouse_y_window / window_height * HEIGHT

        # Read calibration state (read once before drawing)
        calibrated = False
        detections = []
        hand_landmarks = []

        if state_file.exists():
            try:
                with open(state_file, 'rb') as f:
                    state = pickle.load(f)
                    calibrated = state.get('calibrated', False)
                    detections = state.get('detections', [])
                    hand_landmarks = state.get('hand_landmarks', [])
            except Exception as e:
                calibrated = False
                detections = []
                hand_landmarks = []

        # Only allow dragging when not calibrated
        if not calibrated:
            if is_mouse_button_pressed(MOUSE_LEFT_BUTTON):
                # Check which dot was clicked (use window coordinates for hit detection)
                for dot_name, pos in dot_positions.items():
                    dx = mouse_x_window - pos[0]
                    dy = mouse_y_window - pos[1]
                    if (dx * dx + dy * dy) <= (dot_radius * dot_radius):
                        dragging_dot = dot_name
                        break

            if is_mouse_button_down(MOUSE_LEFT_BUTTON) and dragging_dot:
                # Store position in 1920x1080 space and as relative
                dot_positions_relative[dragging_dot] = [
                    mouse_x / WIDTH,
                    mouse_y / HEIGHT
                ]

            if is_mouse_button_released(MOUSE_LEFT_BUTTON):
                if dragging_dot:
                    # Save relative positions
                    try:
                        with open(dots_file, 'wb') as f:
                            pickle.dump(dot_positions_relative, f)
                    except:
                        pass
                dragging_dot = None

        begin_drawing()
        clear_background(BLACK)

        if not calibrated:
            # Show calibration dots at custom positions (draggable)
            magenta = Color(255, 0, 255, 255)

            dot_colors = {
                'red': RED,
                'green': GREEN,
                'magenta': magenta,
                'yellow': YELLOW
            }

            dot_labels = {
                'red': b"RED",
                'green': b"GREEN",
                'magenta': b"MAGENTA",
                'yellow': b"YELLOW"
            }

            # Draw grid mesh connecting the dots
            red_pos = dot_positions['red']
            green_pos = dot_positions['green']
            magenta_pos = dot_positions['magenta']
            yellow_pos = dot_positions['yellow']

            # Draw grid lines
            grid_color = Color(100, 100, 100, 255)

            # Outer rectangle
            draw_line(red_pos[0], red_pos[1], green_pos[0], green_pos[1], grid_color)     # top
            draw_line(green_pos[0], green_pos[1], magenta_pos[0], magenta_pos[1], grid_color)  # right
            draw_line(magenta_pos[0], magenta_pos[1], yellow_pos[0], yellow_pos[1], grid_color)  # bottom
            draw_line(yellow_pos[0], yellow_pos[1], red_pos[0], red_pos[1], grid_color)  # left

            # Diagonals
            draw_line(red_pos[0], red_pos[1], magenta_pos[0], magenta_pos[1], grid_color)
            draw_line(green_pos[0], green_pos[1], yellow_pos[0], yellow_pos[1], grid_color)

            # Draw each dot
            for dot_name, pos in dot_positions.items():
                color = dot_colors[dot_name]
                x, y = pos[0], pos[1]

                # Highlight if being dragged
                if dragging_dot == dot_name:
                    draw_circle(x, y, dot_radius + 10, Color(255, 255, 255, 100))

                draw_circle(x, y, dot_radius, color)
                draw_circle_lines(x, y, dot_radius + 5, WHITE)

                # Draw label
                label = dot_labels[dot_name]
                label_width = len(label) * 8
                draw_text(label, x - label_width // 2, y + dot_radius + 15, 20, WHITE)

            # Draw instructions in center (scaled to window)
            draw_text(b"CALIBRATION MODE - Drag dots to position", window_width // 2 - 300, window_height // 2 - 50, 30, WHITE)
            draw_text(b"Press F11 for fullscreen", window_width // 2 - 150, window_height // 2, 20, LIGHTGRAY)
            draw_text(b"Point camera at screen and press SPACE in Dev Window", window_width // 2 - 350, window_height // 2 + 30, 20, LIGHTGRAY)
        else:
            # Show warped camera feed
            if frame_file.exists():
                try:
                    warped_frame = np.load(frame_file)
                    warped_texture = create_texture_from_frame(warped_frame)

                    if warped_texture:
                        draw_texture(warped_texture, 0, 0, WHITE)

                        # Draw hand landmarks with skeletal structure
                        for hand in hand_landmarks:
                            # Draw connections (skeletal lines)
                            for connection in HAND_CONNECTIONS:
                                start_idx, end_idx = connection
                                if start_idx < len(hand) and end_idx < len(hand):
                                    start_pos = hand[start_idx]
                                    end_pos = hand[end_idx]
                                    draw_line(start_pos[0], start_pos[1],
                                             end_pos[0], end_pos[1],
                                             Color(0, 255, 255, 255))  # Cyan lines

                            # Draw landmark dots on top of lines
                            for i, (x, y) in enumerate(hand):
                                # Different colors for different parts
                                if i == 0:  # Wrist
                                    color = Color(255, 0, 0, 255)  # Red
                                    radius = 8
                                elif i in [4, 8, 12, 16, 20]:  # Fingertips
                                    color = Color(0, 255, 0, 255)  # Green
                                    radius = 6
                                else:  # Other joints
                                    color = Color(255, 255, 0, 255)  # Yellow
                                    radius = 4

                                draw_circle(x, y, radius, color)
                                draw_circle_lines(x, y, radius + 2, WHITE)

                        # Draw bounding boxes (if needed for object detection)
                        for detection in detections:
                            x1, y1, x2, y2 = detection['bbox']
                            conf = detection['confidence']
                            class_name = detection['class']

                            draw_rectangle_lines(x1, y1, x2 - x1, y2 - y1, GREEN)
                            label = f"{class_name} {conf:.2f}"
                            draw_text(label.encode(), x1, y1 - 20, 20, GREEN)

                        unload_texture(warped_texture)
                except Exception as e:
                    draw_text(f"Error: {str(e)}".encode(), 10, 10, 20, RED)

        end_drawing()

    close_window()


if __name__ == "__main__":
    main()
