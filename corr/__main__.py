import corr.image.util as util

from skimage import color
from skimage import data
from skimage import io

import numpy as np


def main() -> None:
    gray = data.camera()

    presentation = util.rbg_presentation_image(gray)
    util.draw_rectangle(presentation, (0.1, 0.1), (0.3, 0.3), (0, 255, 0))
    util.draw_rectangle(presentation, (0.15, 0.15), (0.35, 0.35), (255, 0, 0))

    io.imshow(presentation)
    io.show()


if __name__ == '__main__':
    main()
