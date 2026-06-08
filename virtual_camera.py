import pyvirtualcam
from pyvirtualcam import PixelFormat


def start_virtual_camera(width, height, fps):
    """
    Starts a virtual camera.
    OBS virtual camera must NOT already be running,
    otherwise pyvirtualcam cannot use it.
    """

    cam = pyvirtualcam.Camera(
        width=width,
        height=height,
        fps=fps,
        fmt=PixelFormat.BGR,
        backend="obs"
    )

    print(f"Virtual camera started: {cam.device}")

    return cam