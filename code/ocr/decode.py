import cv2
import matplotlib.pyplot as plt
import numpy as np
from skimage import measure

from generate_samples import get_char_png
from ocr_train import get_unet
from skimage.morphology import square, dilation, erosion

if __name__ == "__main__":
    model_h5 = "ocr.h5"

    print("Model : %s" % model_h5)

    model = get_unet()

    model.load_weights(model_h5, by_name=True)

    # img, mask = get_char_png("ttf")
    # mask = mask.squeeze()

    img = cv2.imread("example6.png")
    img = cv2.resize(img, (256, 256))

    mask_pred = (
        model.predict(img[np.newaxis, ...].astype(np.float)).argmax(axis=-1).squeeze()
    )

    show_mask = mask_pred

    show_mask = erosion(show_mask, square(3))
    show_mask = dilation(show_mask, square(3))

    label_image = measure.label(show_mask, background=0, connectivity=2)

    all_chars = []

    fig, ax = plt.subplots(figsize=(15, 15))
    ax.imshow(img, cmap=plt.cm.gray)

    for region in measure.regionprops(label_image):
        # take regions with large enough areas
        if region.area >= 40:
            # draw rectangle around segmented coins

            minr, minc, maxr, maxc = region.bbox

            occurances = [x for x in show_mask[minr:maxr, minc:maxc].ravel() if x != 0]

            char_idx = max(set(occurances), key=occurances.count)

            all_chars.append(
                {"char": str(char_idx), "x0": minc, "x1": maxc, "y0": minr, "y1": maxr}
            )

            bx = (minc, maxc, maxc, minc, minc)
            by = (minr, minr, maxr, maxr, minr)

            ax.plot(bx, by, "-b", linewidth=1)
            ax.text(minc, minr, str(char_idx))

    plt.show()

    print(all_chars)