# part of this script was taken from https://github.com/jocicmarko/ultrasound-nerve-segmentation

from pathlib import Path

import imgaug.augmenters as iaa
import numpy as np
from tensorflow.keras import losses
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import Input, MaxPooling2D
from tensorflow.keras.layers import (
    concatenate,
    Conv2D,
    Conv2DTranspose,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from ocr.generate_samples import get_grid_char_img

batch_size = 32
input_shape = (256, 256)


def get_detector():
    i = 2
    inputs = Input((None, None, 3))

    conv1 = Conv2D(2 ** i, 3, padding="same", activation="selu")(inputs)
    conv1 = Conv2D(2 ** i, 3, padding="same", activation="selu")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(2 * 2 ** i, 3, padding="same", activation="selu")(pool1)
    conv2 = Conv2D(2 * 2 ** i, 3, padding="same", activation="selu")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(4 * 2 ** i, 3, padding="same", activation="selu")(pool2)
    conv3 = Conv2D(4 * 2 ** i, 3, padding="same", activation="selu")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(8 * 2 ** i, 3, padding="same", activation="selu")(pool3)
    conv4 = Conv2D(8 * 2 ** i, 3, padding="same", activation="selu")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(16 * 2 ** i, 3, padding="same", activation="selu")(pool4)
    conv5 = Conv2D(16 * 2 ** i, 3, padding="same", activation="selu")(conv5)

    up6 = concatenate(
        [
            Conv2DTranspose(
                16 * 2 ** i, 2, strides=2, padding="same", activation="selu"
            )(conv5),
            conv4,
        ],
        axis=3,
    )
    conv6 = Conv2D(8 * 2 ** i, (3, 3), padding="same", activation="selu")(up6)
    conv6 = Conv2D(8 * 2 ** i, (3, 3), padding="same", activation="selu")(conv6)

    up7 = concatenate(
        [
            Conv2DTranspose(
                8 * 2 ** i, 2, strides=2, padding="same", activation="selu"
            )(conv6),
            conv3,
        ],
        axis=3,
    )
    conv7 = Conv2D(4 * 2 ** i, 3, padding="same", activation="selu")(up7)
    conv7 = Conv2D(4 * 2 ** i, 3, padding="same", activation="selu")(conv7)

    up8 = concatenate(
        [
            Conv2DTranspose(
                4 * 2 ** i, 2, strides=2, padding="same", activation="selu"
            )(conv7),
            conv2,
        ],
        axis=3,
    )
    conv8 = Conv2D(3 * 2 ** i, (3, 3), padding="same", activation="selu")(up8)
    conv8 = Conv2D(3 * 2 ** i, (3, 3), padding="same", activation="selu")(conv8)

    up9 = concatenate(
        [
            Conv2DTranspose(
                2 * 2 ** i, 2, strides=2, padding="same", activation="selu"
            )(conv8),
            conv1,
        ],
        axis=3,
    )
    conv9 = Conv2D(2 * 2 ** i, (3, 3), padding="same", activation="selu")(up9)
    conv9 = Conv2D(2 * 2 ** i, (3, 3), padding="same", activation="selu")(conv9)

    conv10 = Conv2D(1, (1, 1), activation="sigmoid")(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(
        optimizer=Adam(lr=3e-4), loss=losses.binary_crossentropy, metrics=["acc"],
    )

    return model


def get_seq():
    sometimes = lambda aug: iaa.Sometimes(0.1, aug)
    seq = iaa.Sequential(
        [
            sometimes(iaa.AdditiveGaussianNoise(scale=0.07 * 255)),
            sometimes(iaa.GaussianBlur(sigma=(0, 3.0))),
            sometimes(iaa.MedianBlur(k=(1, 5))),
            sometimes(iaa.AverageBlur(k=((5, 11), (1, 3)))),
            sometimes(iaa.AveragePooling([2, 8])),
            sometimes(iaa.MaxPooling([2, 8])),
            sometimes(
                iaa.Sequential(
                    [
                        iaa.Resize({"height": 64, "width": 64}),
                        iaa.Resize({"height": input_shape[0], "width": input_shape[1]}),
                    ]
                )
            ),
            sometimes(
                iaa.Sequential(
                    [
                        iaa.Resize({"height": 512, "width": 512}),
                        iaa.Resize({"height": input_shape[0], "width": input_shape[1]}),
                    ]
                )
            ),
        ],
        random_order=True,
    )
    return seq


def gen(size=8, fonts_path="ttf", augment=True):
    seq = get_seq()

    fonts_paths = [str(x) for x in Path(fonts_path).glob("*.otf")] + [
        str(x) for x in Path(fonts_path).glob("*.ttf")
    ]

    while True:

        samples = [get_grid_char_img(fonts_paths=fonts_paths) for _ in range(size)]
        list_images, list_gt = zip(*samples)

        if augment:
            list_images = seq.augment_images(images=list_images)

        array_gt = np.array(list_gt).squeeze()
        binary_gt = (array_gt > 0).astype(float)

        yield np.array(list_images), binary_gt


def train_detector():
    model_h5 = "ocr_detector.h5"

    print("Model : %s" % model_h5)

    model = get_detector()

    try:
        model.load_weights(model_h5, by_name=True)
    except:
        pass

    checkpoint = ModelCheckpoint(
        model_h5, monitor="val_acc", verbose=1, save_best_only=True, mode="max"
    )
    early = EarlyStopping(monitor="val_acc", mode="max", patience=40, verbose=1)
    redonplat = ReduceLROnPlateau(
        monitor="val_acc", mode="max", patience=20, verbose=1, min_lr=1e-7
    )
    callbacks_list = [checkpoint, early, redonplat]

    history = model.fit_generator(
        gen(),
        epochs=100,
        verbose=1,
        steps_per_epoch=128,
        validation_data=gen(augment=False),
        validation_steps=64,
        callbacks=callbacks_list,
        # use_multiprocessing=True,
        # workers=8,
    )

    model.save_weights(model_h5)


if __name__ == "__main__":
    train_detector()
