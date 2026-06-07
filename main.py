import cv2
from image_stats import calculate_basic_stats


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        stats = calculate_basic_stats(frame)

        # Print statistics for each RGB channel
        print("Red:", stats["Red"])
        print("Green:", stats["Green"])
        print("Blue:", stats["Blue"])
        print("-" * 50)

        cv2.imshow("Original Webcam", frame)

        if cv2.waitKey(1) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()