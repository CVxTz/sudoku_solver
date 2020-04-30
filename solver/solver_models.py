from tensorflow.keras.layers import (
    Embedding,
    GRU,
    Bidirectional,
    Input,
    SpatialDropout1D,
    Dense,
    Add,
    GaussianNoise,
)
from tensorflow.keras.models import Model
from tensorflow.keras.losses import sparse_categorical_crossentropy
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.engine.base_layer import Layer
import numpy as np
from generator.Generator import Generator


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
    input_ = Input((None,))
    input_pos = Input((None,))

    x = Embedding(10, 10)(input_)
    x_pos = Embedding(100, 10)(input_pos)

    x = TTGaussianNoise(stddev=0.3)(x)

    x = Add()([x, x_pos])

    x = Bidirectional(GRU(256, return_sequences=True))(x)
    x = Bidirectional(GRU(256, return_sequences=True))(x)

    out = Dense(10, activation="softmax")(x)

    model = Model(inputs=[input_, input_pos], outputs=out)

    model.compile(
        loss=sparse_categorical_crossentropy, optimizer="Adam", metrics=["acc"]
    )

    model.summary()

    return model


def predict(arr, model):
    batch_size = 64
    x_in = [arr] * batch_size
    x_pos = [list(range(81))] * batch_size

    X_in = np.array(x_in)
    X_pos = np.array(x_pos)

    pred = model.predict([X_in, X_pos])
    pred = pred.argmax(axis=-1)

    pred_gens = [Generator(pred[i, ...].ravel().tolist()) for i in range(32)]

    for pred_gen in pred_gens:
        if pred_gen.board.is_solved():
            return pred_gen

    return pred_gens[0]

