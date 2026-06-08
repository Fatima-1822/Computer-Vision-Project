import cv2


def linear_transformation(frame, alpha=1.2, beta=30):
    """
    Applies linear transformation to the image.

    Formula:
    new_pixel = alpha * old_pixel + beta

    alpha controls contrast.
    beta controls brightness.
    """

    transformed_frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    return transformed_frame