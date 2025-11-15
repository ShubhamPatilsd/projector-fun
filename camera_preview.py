"""Camera preview window - shows raw camera feed"""
import cv2
import numpy as np
from pathlib import Path
import time


def main():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Camera Preview Window - Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Display the frame
        cv2.imshow('Camera Preview', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
