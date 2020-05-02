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