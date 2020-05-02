import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure

from ocr.ocr_detector import get_detector
from ocr.ocr_recognizer import get_recognizer


def mask_to_bboxes(show_mask, threshold=40):
    label_image = measure.label(show_mask, background=0, connectivity=2)

    all_chars = []

    for region in measure.regionprops(label_image):
        if region.area >= threshold:
            minr, minc, maxr, maxc = region.bbox

            all_chars.append(
                {"char": None, "minc": minc, "maxc": maxc, "minr": minr, "maxr": maxr}
            )

    return all_chars


def predict_mask(detector_model, img):
    mask_pred = detector_model.predict(img[np.newaxis, ...].astype(np.float))
    mask_pred = (mask_pred > 0.5).astype(np.int).squeeze()
    return mask_pred


def predict_char_class(recognizer_model, img, all_chars):
    list_bboxes = []

    for char in all_chars:
        minr, minc, maxr, maxc = char["minr"], char["minc"], char["maxr"], char["maxc"]
        size = max(maxr - minr, maxc - minc)

        list_bboxes.append(img[minr : (minr + size), minc : (minc + size), :])

    list_bboxes = [cv2.resize(x, (32, 32)) for x in list_bboxes]

    array_bboxes = np.array(list_bboxes).astype(np.float)

    preds = recognizer_model.predict(array_bboxes).argmax(axis=-1).ravel().tolist()

    # list_bboxes = [cv2.imwrite("%s_%s.png" % (j, i), x) for j, (i, x) in enumerate(zip(preds, list_bboxes))]

    for char, pred in zip(all_chars, preds):
        char["char"] = pred

    return all_chars


def bucket_l(l, cutoff=10):
    res = [[]]

    for x in l:
        if len(res[-1]) == 0 or abs(res[-1][-1] - x) < cutoff:
            res[-1].append(x)
        else:
            res.append([x])

    return res


def infer_rows_and_cols(chars):
    cutoff = 10

    row = [(x["maxr"] + x["minr"]) / 2 for x in chars]
    col = [(x["maxc"] + x["minc"]) / 2 for x in chars]

    row = sorted(row)
    col = sorted(col)

    row = bucket_l(row, cutoff=cutoff)
    col = bucket_l(col, cutoff=cutoff)

    row = [np.median(x) for x in row]
    col = [np.median(x) for x in col]

    grid = []

    for r in row:
        r_i = []
        for c in col:
            char = 0
            for c_char in chars:
                if (
                    abs((c_char["minc"] + c_char["maxc"]) / 2 - c)
                    + abs((c_char["minr"] + c_char["maxr"]) / 2 - r)
                ) < 2 * cutoff:
                    char = c_char["char"]
                    break
            r_i.append(char)
        grid.append(r_i)

    return grid


def img_to_grid(
    img, detector_model, recognizer_model, plot_path=None, print_result=False
):
    img = cv2.resize(img, (256, 256))

    show_mask = predict_mask(detector_model, img)

    all_chars = mask_to_bboxes(show_mask)

    all_chars = predict_char_class(recognizer_model, img, all_chars)

    if plot_path is not None:

        fig, ax = plt.subplots(figsize=(15, 15))
        plt.axis("off")

        ax.imshow(img, cmap=plt.cm.gray)

        for char in all_chars:
            minr, minc, maxr, maxc = (
                char["minr"],
                char["minc"],
                char["maxr"],
                char["maxc"],
            )

            bx = (minc, maxc, maxc, minc, minc)
            by = (minr, minr, maxr, maxr, minr)

            ax.plot(bx, by, "-b", linewidth=1)
            ax.text(minc, minr, str(char["char"]), fontsize=24)

        plt.savefig(plot_path, bbox_inches="tight", pad_inches=0)

    grid = infer_rows_and_cols(all_chars)

    if print_result:
        for r in grid:
            print(r)

    return grid


if __name__ == "__main__":
    detector_model_h5 = "ocr_detector.h5"
    detector_model = get_detector()
    detector_model.load_weights(detector_model_h5)

    recognizer_model_h5 = "ocr_recognizer.h5"
    recognizer_model = get_recognizer()
    recognizer_model.load_weights(recognizer_model_h5)

    img = cv2.imread("example6.png")

    img_to_grid(
        img, detector_model, recognizer_model, plot_path="plot.png", print_result=False
    )
