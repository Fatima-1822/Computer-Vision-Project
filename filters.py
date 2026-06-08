import cv2
import numpy as np


def sobel_edge_detection(frame):
    """
    Applies Sobel edge detection.

    Sobel detects strong changes in pixel intensity.
    These changes usually represent edges.
    """

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    sobel_combined = cv2.magnitude(sobel_x, sobel_y)

    sobel_combined = np.uint8(np.clip(sobel_combined, 0, 255))

    sobel_bgr = cv2.cvtColor(sobel_combined, cv2.COLOR_GRAY2BGR)

    return sobel_bgr