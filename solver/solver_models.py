import numpy as np
from tensorflow.keras.layers import (
    Input,
    Add,
    Convolution2D,
)
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import array_ops

from generator.Generator import Generator
from solver.utils import binarize_along_last_axis


class TTGaussianNoise(Layer):
    def __init__(self, stddev, **kwargs):
        super(TTGaussianNoise, self).__init__(**kwargs)
        self.supports_masking = True
        self.stddev = stddev

    def call(self, inputs):
        return inputs + K.random_normal(
            shape=array_ops.shape(inputs), mean=0.0, stddev=self.stddev
        )

    def get_config(self):
        config = {"stddev": self.stddev}
        base_config = super(TTGaussianNoise, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


def get_model():
    filter_number = 512

    input_ = Input((9, 9, 10))

    a1 = Convolution2D(filter_number, kernel_size=3, activation="selu", padding="same")(input_)
    a2 = Convolution2D(filter_number, kernel_size=3, activation="selu", padding="same", dilation_rate=2)(input_)
    a3 = Convolution2D(filter_number, kernel_size=3, activation="selu", padding="same", dilation_rate=4)(input_)
    a4 = Convolution2D(filter_number, kernel_size=3, activation="selu", padding="same", dilation_rate=8)(input_)
    x1 = Add()([a1, a2, a3, a4])

    x2 = Convolution2D(filter_number, kernel_size=3, activation="selu", padding="same")(x1)
    x2 = Add()([x1, x2])

    x3 = Convolution2D(filter_number, kernel_size=3, activation="selu", padding="same")(x2)

    x4 = Convolution2D(filter_number, kernel_size=3, activation="selu", padding="same")(x3)
    x4 = Add()([x1, x4])

    x5 = Convolution2D(filter_number, kernel_size=3, activation="selu", padding="same")(x4)

    x6 = Convolution2D(filter_number, kernel_size=3, activation="selu", padding="same")(x5)
    x6 = Add()([x1, x6])

    x7 = Convolution2D(filter_number, kernel_size=3, activation="selu", padding="same")(x6)
    x8 = Convolution2D(filter_number, kernel_size=3, activation="selu", padding="same")(x7)

    x9 = Convolution2D(10, kernel_size=3, activation="softmax", padding="same")(x8)

    model = Model(inputs=input_, outputs=x9)

    model.compile(
        loss=categorical_crossentropy, optimizer=Adam(learning_rate=1e-4), metrics=["acc"]
    )

    model.summary()

    return model


def predict(arr, model):
    x_in = [arr]

    X_in = np.array(x_in)
    X_in = binarize_along_last_axis(X_in, n_classes=10)

    pred = model.predict(X_in)
    pred = pred.argmax(axis=-1)

    pred_gen = Generator(pred[0, ...].ravel().tolist())

    return pred_gen


def predict_sequential(arr, model):

    X_in = np.array(arr)

    while np.sum(X_in == 0):
        X = X_in[np.newaxis, ...]
        X = binarize_along_last_axis(X, n_classes=10)

        pred = model.predict(X).squeeze()

        pred_max = pred.max(axis=-1)
        pred_argmax = pred.argmax(axis=-1)

        i_all, j_all = np.where(X_in == 0)
        max_idx = pred_max[X_in == 0].argmax()
        i, j = i_all[max_idx], j_all[max_idx]

        X_in[i, j] = pred_argmax[i, j]

    sodoku_gen = Generator(X_in.ravel().tolist())

    return sodoku_gen


if __name__ == "__main__":

    arr = np.array([[1, 2, 3], [1, 0, 3], [0, 0, 3]])
    pred = np.array([[[9.1, 12.9, 0.3], [5.15, 111.9, 0.3], [7.1, 3.9, 5.3]],
                     [[0.1, 0.9, 0.3], [0.1, 5.9, 7.3], [0.1, 9.9, 0.3]],
                     [[110.1, 0.9, 0.3], [0.1, 2.9, 0.3], [0.1, 0.9, 0.3]]])

    pred_max = pred.max(axis=-1)
    pred_argmax = pred.argmax(axis=-1)
    print(arr)
    print(pred_max)
    print(pred_argmax)

    max_value = pred_max[arr == 0].max()
    max_idx = pred_max[arr == 0].argmax()

    i, j = np.where(arr == 0)
    print(i)
    print(j)

    print(max_value)
    print(max_idx)
