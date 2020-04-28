from random import choice

import cv2
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import time
from skimage.morphology import square, dilation
from glob import glob
from pathlib import Path


def get_char_png(fonts_path):

    paths = [str(x) for x in Path(fonts_path).glob("*.otf")]
    font_name = choice(paths)
    size = np.random.randint(5, 50)
    font = ImageFont.truetype(font_name, size)

    image_size = 256

    img = 255 * np.ones((image_size, image_size, 3), np.uint8)
    mask = np.zeros((image_size, image_size, 1), np.uint8)

    pil_img = Image.fromarray(img)
    draw = ImageDraw.Draw(pil_img)
    step = min(image_size // size, 30)
    for i in range(step):
        for j in range(step):

            start_x = i * size
            start_y = j * size

            if np.random.uniform(0, 1) < 0.2:
                integer = choice([1, 2, 3, 4, 5, 6, 7, 8, 9])

                color = (0, 0, 0) if np.random.uniform(0, 1) < 0.9 else \
                    (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

                draw.text((start_x, start_y), str(integer), color, font=font)

                img = np.array(pil_img)

                char_box = (
                    255 - img[start_y : (start_y + size), start_x : (start_x + size), :]
                )

                char_box = cv2.GaussianBlur(char_box, (5, 5), cv2.BORDER_DEFAULT)

                char_mask = mask[
                    start_y : (start_y + size), start_x : (start_x + size), :
                ]

                char_mask[char_box[..., 0] > 10] = integer

    for i in range(step):
        for j in range(step):

            start_x = i * size
            start_y = j * size

            if np.random.uniform(0, 1) < 0.05:
                draw.line((start_x, 0, start_x, image_size),
                          fill=(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)),
                          width=np.random.randint(0, 5))

            if np.random.uniform(0, 1) < 0.05:
                draw.line((0, start_y, image_size, start_y),
                          fill=(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)),
                          width=np.random.randint(0, 5))

    img = np.array(pil_img)

    for i in range(3):
        img[..., i] = np.random.uniform(0.5, 1) * img[..., i]

    return img, mask


if __name__ == "__main__":

    start = time.time()

    img, mask = get_char_png(fonts_path="ttf")

    cv2.imwrite("img.png", img)
    cv2.imwrite("mask.png", 30 * mask)

    print(time.time() - start)
