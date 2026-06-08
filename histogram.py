import cv2
import numpy as np


def create_rgb_histogram_image(frame):
    """
    Creates a histogram image for the RGB channels.
    OpenCV uses BGR, so we handle channels carefully.

    The output is an image showing three histogram lines:
    Red, Green, Blue.
    """

    hist_height = 300
    hist_width = 512
    bin_width = hist_width // 256

    histogram_image = np.zeros((hist_height, hist_width, 3), dtype=np.uint8)

    channels = {
        "Blue": (0, (255, 0, 0)),
        "Green": (1, (0, 255, 0)),
        "Red": (2, (0, 0, 255)),
    }

    for channel_name, (channel_index, color) in channels.items():
        histogram = cv2.calcHist(
            [frame],
            [channel_index],
            None,
            [256],
            [0, 256]
        )

        cv2.normalize(histogram, histogram, 0, hist_height, cv2.NORM_MINMAX)

        for x in range(1, 256):
            x1 = bin_width * (x - 1)
            y1 = hist_height - int(histogram[x - 1][0])

            x2 = bin_width * x
            y2 = hist_height - int(histogram[x][0])

            cv2.line(histogram_image, (x1, y1), (x2, y2), color, 1)

    return histogram_image
def equalize_histogram(frame):
    """
    Applies histogram equalization to improve contrast.

    We convert BGR to YCrCb.
    Then we equalize only the Y channel because Y represents brightness.
    This keeps the colors more natural.
    """

    ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)

    y_channel, cr_channel, cb_channel = cv2.split(ycrcb)

    equalized_y = cv2.equalizeHist(y_channel)

    equalized_ycrcb = cv2.merge((equalized_y, cr_channel, cb_channel))

    equalized_frame = cv2.cvtColor(equalized_ycrcb, cv2.COLOR_YCrCb2BGR)

    return equalized_frame