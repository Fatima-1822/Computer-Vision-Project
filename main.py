import cv2

from image_stats import calculate_basic_stats, calculate_entropy
from transformation import linear_transformation
from histogram import create_rgb_histogram_image, equalize_histogram
from filters import sobel_edge_detection


def draw_stats_on_frame(frame, stats):
    """
    Draw basic statistics on the image.
    """

    y = 30

    for channel_name in ["Red", "Green", "Blue"]:
        channel_stats = stats[channel_name]

        text = (
            f"{channel_name}: "
            f"Mean={channel_stats['mean']:.1f}, "
            f"Mode={channel_stats['mode']}, "
            f"Std={channel_stats['std']:.1f}, "
            f"Min={channel_stats['min']}, "
            f"Max={channel_stats['max']}"
        )

        cv2.putText(
            frame,
            text,
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        y += 25

    return frame


def main():

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    show_transformed = False
    show_equalized = False
    show_sobel = False

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Select processing mode
        if show_sobel:
            output_frame = sobel_edge_detection(frame)

        elif show_equalized:
            output_frame = equalize_histogram(frame)

        elif show_transformed:
            output_frame = linear_transformation(frame)

        else:
            output_frame = frame.copy()

        # Basic statistics
        stats = calculate_basic_stats(output_frame)
        output_frame = draw_stats_on_frame(output_frame, stats)

        # Entropy
        entropy = calculate_entropy(output_frame)

        cv2.putText(
            output_frame,
            f"Entropy: {entropy:.2f}",
            (20, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        # Controls help text
        cv2.putText(
            output_frame,
            "t=Linear  e=Equalization  s=Sobel  q=Quit",
            (20, output_frame.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        # Main webcam window
        cv2.imshow("Computer Vision Project", output_frame)

        # Histogram window
        histogram_image = create_rgb_histogram_image(output_frame)
        cv2.imshow("RGB Histogram", histogram_image)

        key = cv2.waitKey(1)

        # Linear transformation
        if key == ord("t"):
            show_transformed = not show_transformed
            show_equalized = False
            show_sobel = False

        # Histogram equalization
        if key == ord("e"):
            show_equalized = not show_equalized
            show_transformed = False
            show_sobel = False

        # Sobel filter
        if key == ord("s"):
            show_sobel = not show_sobel
            show_transformed = False
            show_equalized = False

        # Quit
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()