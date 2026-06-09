import cv2

from image_stats import calculate_basic_stats, calculate_entropy
from transformation import linear_transformation
from histogram import create_rgb_histogram_image, equalize_histogram
from filters import sobel_edge_detection
from virtual_camera import start_virtual_camera
from special_task import FunnyFaceEmojiTask


def draw_stats_on_frame(frame, stats, entropy):
    """
    Draw a clean frontend/dashboard on the camera frame.
    Shows RGB statistics, entropy, and project menu.
    """

    h, w, _ = frame.shape

    # -----------------------------
    # Top title bar
    # -----------------------------
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 90), (15, 15, 30), -1)
    frame = cv2.addWeighted(overlay, 0.80, frame, 0.20, 0)

    cv2.putText(
        frame,
        "Computer Vision Virtual Camera Project",
        (25, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "Live OpenCV Processing + Image Statistics + Virtual Camera Output",
        (25, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # -----------------------------
    # Left statistics panel
    # -----------------------------
    panel_x = 20
    panel_y = 115
    panel_w = 420
    panel_h = 260

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (20, 20, 35),
        -1,
    )
    frame = cv2.addWeighted(overlay, 0.78, frame, 0.22, 0)

    cv2.rectangle(
        frame,
        (panel_x, panel_y),
        (panel_x + panel_w, panel_y + panel_h),
        (0, 255, 255),
        2,
    )

    cv2.putText(
        frame,
        "RGB CHANNEL STATISTICS",
        (panel_x + 15, panel_y + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Table headers
    start_y = panel_y + 65

    cv2.putText(
        frame,
        "CH",
        (panel_x + 15, start_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )

    cv2.putText(
        frame,
        "Mean",
        (panel_x + 60, start_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )

    cv2.putText(
        frame,
        "Mode",
        (panel_x + 150, start_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )

    cv2.putText(
        frame,
        "Std",
        (panel_x + 230, start_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )

    cv2.putText(
        frame,
        "Min/Max",
        (panel_x + 300, start_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
    )

    cv2.line(
        frame,
        (panel_x + 15, start_y + 10),
        (panel_x + panel_w - 15, start_y + 10),
        (100, 100, 100),
        1,
    )

    channel_colors = {
        "Red": (80, 80, 255),
        "Green": (80, 255, 80),
        "Blue": (255, 120, 80),
    }

    row_y = start_y + 40

    for channel_name in ["Red", "Green", "Blue"]:
        channel_stats = stats[channel_name]
        color = channel_colors[channel_name]

        cv2.putText(
            frame,
            channel_name[0],
            (panel_x + 18, row_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

        cv2.putText(
            frame,
            f"{channel_stats['mean']:.1f}",
            (panel_x + 60, row_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

        cv2.putText(
            frame,
            f"{channel_stats['mode']}",
            (panel_x + 150, row_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

        cv2.putText(
            frame,
            f"{channel_stats['std']:.1f}",
            (panel_x + 230, row_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

        cv2.putText(
            frame,
            f"{channel_stats['min']}/{channel_stats['max']}",
            (panel_x + 300, row_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
        )

        row_y += 45

    # -----------------------------
    # Entropy box
    # -----------------------------
    entropy_x = panel_x + 15
    entropy_y = panel_y + 215

    cv2.rectangle(
        frame,
        (entropy_x, entropy_y),
        (entropy_x + 385, entropy_y + 32),
        (35, 35, 55),
        -1,
    )

    cv2.putText(
        frame,
        f"Entropy / Image Information: {entropy:.2f}",
        (entropy_x + 10, entropy_y + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    # -----------------------------
    # Bottom menu bar
    # -----------------------------
    menu_h = 75

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - menu_h), (w, h), (15, 15, 30), -1)
    frame = cv2.addWeighted(overlay, 0.82, frame, 0.18, 0)

    cv2.line(frame, (0, h - menu_h), (w, h - menu_h), (0, 255, 255), 2)

    cv2.putText(
        frame,
        "MENU",
        (25, h - 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    controls = [
        ("T", "Linear"),
        ("E", "Equalize"),
        ("S", "Sobel"),
        ("F", "Emoji"),
        ("Q", "Quit"),
    ]

    x_pos = 110

    for key, label in controls:
        cv2.rectangle(
            frame,
            (x_pos, h - 58),
            (x_pos + 38, h - 25),
            (0, 255, 255),
            2,
        )

        cv2.putText(
            frame,
            key,
            (x_pos + 10, h - 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.putText(
            frame,
            label,
            (x_pos + 48, h - 36),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

        x_pos += 150

    return frame


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Getting webcam size because virtual camera needs same width and height
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Simple fixed FPS for virtual camera output
    fps = 20

    # Create special task object for funny face / emoji mode
    special_task = FunnyFaceEmojiTask()

    # Start virtual camera so OBS/Zoom can receive our processed frame
    virtual_cam = start_virtual_camera(width, height, fps)

    show_transformed = False
    show_equalized = False
    show_sobel = False
    show_special = False

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Select processing mode
        if show_special:
            output_frame = special_task.process_frame(frame)

        elif show_sobel:
            output_frame = sobel_edge_detection(frame)

        elif show_equalized:
            output_frame = equalize_histogram(frame)

        elif show_transformed:
            output_frame = linear_transformation(frame)

        else:
            output_frame = frame.copy()

        # Basic statistics
        stats = calculate_basic_stats(output_frame)

        # Entropy
        entropy = calculate_entropy(output_frame)

        # Draw stats frontend only when special mode is OFF
        # Special mode has its own frontend in special_task.py
        if not show_special:
            output_frame = draw_stats_on_frame(output_frame, stats, entropy)

        # Main webcam window
        cv2.imshow("Computer Vision Project", output_frame)

        # Send processed frame to virtual camera for OBS/Zoom/etc.
        virtual_cam.send(output_frame)
        virtual_cam.sleep_until_next_frame()

        # Histogram window
        histogram_image = create_rgb_histogram_image(output_frame)
        cv2.imshow("RGB Histogram", histogram_image)

        # Read keyboard input
        key = cv2.waitKey(1) & 0xFF

        # Send key to special task
        # This allows A/H/X/K/Z/D/C/G/N/L keys to work in emoji mode
        special_task.set_mode_from_key(key)

        # Linear transformation
        if key == ord("t"):
            show_transformed = not show_transformed
            show_equalized = False
            show_sobel = False
            show_special = False

        # Histogram equalization
        if key == ord("e"):
            show_equalized = not show_equalized
            show_transformed = False
            show_sobel = False
            show_special = False

        # Sobel filter
        if key == ord("s"):
            show_sobel = not show_sobel
            show_transformed = False
            show_equalized = False
            show_special = False

        # Funny face / emoji special task
        if key == ord("f"):
            show_special = not show_special
            show_transformed = False
            show_equalized = False
            show_sobel = False
            print("Emoji special mode:", show_special)

        # Quit
        if key == ord("q"):
            break

    cap.release()
    virtual_cam.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()