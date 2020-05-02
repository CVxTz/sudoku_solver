import time
from pathlib import Path
from random import choice

import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from skimage.morphology import square, dilation


def get_grid_char_img(fonts_paths):
    font_name = choice(fonts_paths)
    size = np.random.randint(5, 40)
    font = ImageFont.truetype(font_name, size)

    image_size = 256

    img = 255 * np.ones((image_size, image_size, 3), np.uint8)
    mask = np.zeros((image_size, image_size), np.uint8)

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    step = int(min(image_size / (1.2 * size), 30))
    for i in range(step):
        for j in range(step):

            start_x = int(i * size * 1.2)
            start_y = int(j * size * 1.2)

            if np.random.uniform(0, 1) < 0.2:
                integer = choice([1, 2, 3, 4, 5, 6, 7, 8, 9])

                color = (
                    (0, 0, 0)
                    if np.random.uniform(0, 1) < 0.9
                    else (
                        np.random.randint(0, 255),
                        np.random.randint(0, 255),
                        np.random.randint(0, 255),
                    )
                )

                draw.text((start_x, start_y), str(integer), color, font=font)

                img = np.array(pil_img)

                char_box = (
                    255
                    - img[
                        start_y : int(start_y + 1.2 * size),
                        start_x : int(start_x + 1.2 * size),
                        :,
                    ]
                )
                char_mask = mask[
                    start_y : int(start_y + 1.2 * size),
                    start_x : int(start_x + 1.2 * size),
                ]
                char_mask[char_box[..., 0] > 10] = integer
                mask[
                    start_y : int(start_y + 1.2 * size),
                    start_x : int(start_x + 1.2 * size),
                ] = dilation(char_mask, square(3))

    for i in range(step):
        for j in range(step):

            start_x = int(i * size * 1.2) - 0.3 * size
            start_y = int(j * size * 1.2) + 0.1 * size

            if np.random.uniform(0, 1) < 0.05:
                draw.line(
                    (start_x, 0, start_x, image_size),
                    fill=(
                        np.random.randint(0, 255),
                        np.random.randint(0, 255),
                        np.random.randint(0, 255),
                    ),
                    width=np.random.randint(0, 5),
                )

            if np.random.uniform(0, 1) < 0.05:
                draw.line(
                    (0, start_y, image_size, start_y),
                    fill=(
                        np.random.randint(0, 255),
                        np.random.randint(0, 255),
                        np.random.randint(0, 255),
                    ),
                    width=np.random.randint(0, 5),
                )

    img = np.array(pil_img)

    if np.random.uniform(0, 1) < 0.1:

        for i in range(3):
            img[..., i] = (
                np.clip(1 - np.fabs(np.random.normal(0, 0.2)), 0, 1) * img[..., i]
            )

    return img, mask


def get_char_img(fonts_paths):
    font_name = choice(fonts_paths)
    size = np.random.randint(7, 40)
    font = ImageFont.truetype(font_name, size)

    image_size = int(np.random.uniform(0.9, 1.4) * size)

    img = 255 * np.ones((image_size, image_size, 3), np.uint8)

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)

    integer = choice([1, 2, 3, 4, 5, 6, 7, 8, 9])

    color = (
        (0, 0, 0)
        if np.random.uniform(0, 1) < 0.9
        else (
            np.random.randint(0, 255),
            np.random.randint(0, 255),
            np.random.randint(0, 255),
        )
    )

    draw.text(
        (np.random.randint(0, size // 2 + 1), np.random.randint(0, size // 5)),
        str(integer),
        color,
        font=font,
    )

    img = np.array(pil_img)

    img = cv2.resize(img, (32, 32))

    if np.random.uniform(0, 1) < 0.1:

        for i in range(3):
            img[..., i] = (
                np.clip(1 - np.fabs(np.random.normal(0, 0.2)), 0, 1) * img[..., i]
            )

    return img, integer


if __name__ == "__main__":
    start = time.time()

    fonts_paths = [str(x) for x in Path("ttf").glob("*.otf")] + [
        str(x) for x in Path("ttf").glob("*.ttf")
    ]

    img, mask = get_grid_char_img(fonts_paths=fonts_paths)

    cv2.imwrite("img.png", img)
    cv2.imwrite("mask.png", 30 * mask)

    print(time.time() - start)

    img, label = get_char_img(fonts_paths=fonts_paths)

    cv2.imwrite("char_%s.png" % label, img)
