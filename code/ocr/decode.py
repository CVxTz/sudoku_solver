from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure

from generate_samples import get_grid_char_img
from ocr_detector_train import get_detector
from ocr_recognizer_train import get_recognizer


def mask_to_bboxes(show_mask, threshold=40):
    label_image = measure.label(show_mask, background=0, connectivity=2)

    all_chars = []

    for region in measure.regionprops(label_image):
        if region.area >= threshold:
            minr, minc, maxr, maxc = region.bbox

            all_chars.append({"char": None, "minc": minc, "maxc": maxc, "minr": minr, "maxr": maxr})

    return all_chars


def predict_mask(detector_model, img):
    mask_pred = detector_model.predict(img[np.newaxis, ...].astype(np.float))
    mask_pred = (mask_pred > 0.5).astype(np.int).squeeze()
    return mask_pred


def predict_char_class(recognizer_model, img, all_chars):
    list_bboxes = []

    for char in all_chars:
        minr, minc, maxr, maxc = char['minr'], char['minc'], char['maxr'], char['maxc']
        size = max(maxr - minr, maxc - minc)

        list_bboxes.append(img[minr:(minr + size), minc:(minc + size), :])

    list_bboxes = [cv2.resize(x, (32, 32)) for x in list_bboxes]

    array_bboxes = np.array(list_bboxes).astype(np.float)

    preds = recognizer_model.predict(array_bboxes).argmax(axis=-1).ravel().tolist()

    # list_bboxes = [cv2.imwrite("%s_%s.png" % (j, i), x) for j, (i, x) in enumerate(zip(preds, list_bboxes))]

    for char, pred in zip(all_chars, preds):
        char["char"] = pred

    return all_chars


if __name__ == "__main__":

    detector_model_h5 = "ocr_detector.h5"
    detector_model = get_detector()
    detector_model.load_weights(detector_model_h5)

    recognizer_model_h5 = "ocr_recognizer.h5"
    recognizer_model = get_recognizer()
    recognizer_model.load_weights(recognizer_model_h5)

    fonts_paths = [str(x) for x in Path("ttf").glob("*.otf")] + \
                  [str(x) for x in Path("ttf").glob("*.ttf")]

    # img, mask = get_grid_char_img(fonts_paths)
    # mask = mask.squeeze()

    img = cv2.imread("example3.png")
    img = cv2.resize(img, (256, 256))

    show_mask = predict_mask(detector_model, img)

    all_chars = mask_to_bboxes(show_mask)

    all_chars = predict_char_class(recognizer_model, img, all_chars)

    fig, ax = plt.subplots(figsize=(15, 15))

    ax.imshow(img, cmap=plt.cm.gray)

    for char in all_chars:
        minr, minc, maxr, maxc = char['minr'], char['minc'], char['maxr'], char['maxc']

        bx = (minc, maxc, maxc, minc, minc)
        by = (minr, minr, maxr, maxr, minr)

        ax.plot(bx, by, "-b", linewidth=1)
        ax.text(minc, minr, str(char["char"]))

    plt.show()

    print(all_chars)
