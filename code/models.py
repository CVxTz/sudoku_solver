from tensorflow.keras.layers import Embedding, GRU, Bidirectional, Input, SpatialDropout1D, Dense, Add, Attention
from tensorflow.keras.models import Model
from tensorflow.keras.losses import sparse_categorical_crossentropy


def get_model():
    input_ = Input((None, ))
    input_pos = Input((None, ))

    x = Embedding(10, 10)(input_)
    x_pos = Embedding(100, 10)(input_pos)

    x = SpatialDropout1D(0.1)(x)

    x = Add()([x, x_pos])

    x = Bidirectional(GRU(256, return_sequences=True))(x)
    x = Bidirectional(GRU(256, return_sequences=True))(x)

    out = Dense(10, activation='softmax')(x)

    model = Model(inputs=[input_, input_pos], outputs=out)

    model.compile(loss=sparse_categorical_crossentropy, optimizer="Adam", metrics=['acc'])

    model.summary()

    return model