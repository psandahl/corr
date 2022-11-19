import numpy as np
from skimage import color
from skimage import draw
from skimage import io


def open_grayscale_image(path: str) -> np.ndarray:
    """
    Open an image as grayscale. None if image not is found.

    Parameters:
        path: Path to image.

    Returns:
        Image or None.
    """
    try:
        return io.imread(path, as_gray=True)
    except FileNotFoundError:
        return None


def rbg_presentation_image(image: np.ndarray) -> np.ndarray:
    """
    From a grayscale image make a rgb presentation image.

    Parameters:
        The image.

    Returns:
        The new image.
    """
    return color.gray2rgb(image)


def uv_to_px(image: np.ndarray, uv: tuple) -> tuple():
    """
    Convert from uv to pixel coordinates for the given image.

    Parameters:
        image: The image.
        uv: Tuple u, v.

    Returns:
        The pixel coordinates.
    """
    rows = image.shape[0]
    cols = image.shape[1]

    u, v = uv

    return round(u * (cols - 1)), round(v * (rows - 1))


def px_to_uv(image: np.ndarray, px: tuple) -> tuple():
    """
    Convert from pixel to uv coordinates for the given image.

    Parameters:
        image: The image.
        px: Tuple x, y.

    Returns:
        The uv coordinates.
    """
    rows = image.shape[0]
    cols = image.shape[1]

    x, y = px

    return x / (cols - 1), y / (rows - 1)


def draw_rectangle(image: np.ndarray, uv_ul: tuple, uv_lr: tuple, color: any) -> None:
    """
    Draw a rectangle between the given uv coordinates.

    Parameters:
        image: The image to annotate.
        uv_ul: The uv coordinate for the upper left corner.
        uv_lr: The uv coordinate for the lower right corner.
        color: The color value (shall be given according to image format).

    Returns:
        None.
    """
    start = uv_to_px(image, uv_ul)[::-1]
    end = uv_to_px(image, uv_lr)[::-1]

    rect = draw.rectangle_perimeter(
        start, end=end, shape=image.shape, clip=True)

    draw.set_color(image, rect, color)
