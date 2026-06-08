import numpy as np
from scipy import stats


def calculate_basic_stats(frame):
    """
    Calculates basic statistics for each color channel.
    OpenCV uses BGR order, not RGB.
    """

    channels = {
        "Blue": frame[:, :, 0],
        "Green": frame[:, :, 1],
        "Red": frame[:, :, 2],
    }

    results = {}

    for name, channel in channels.items():
        flat_channel = channel.flatten()

        results[name] = {
            "mean": np.mean(flat_channel),
            "mode": stats.mode(flat_channel, keepdims=True).mode[0],
            "std": np.std(flat_channel),
            "max": np.max(flat_channel),
            "min": np.min(flat_channel),
        }

    return results


def calculate_entropy(frame):
    """
    Calculates entropy of the grayscale version of the frame.
    Entropy measures the amount of information in an image.
    """

    gray_frame = np.mean(frame, axis=2).astype(np.uint8)

    histogram, _ = np.histogram(gray_frame, bins=256, range=(0, 256))

    probabilities = histogram / np.sum(histogram)

    probabilities = probabilities[probabilities > 0]

    entropy = -np.sum(probabilities * np.log2(probabilities))

    return entropy