import sys

import numpy as np

sys.path.append("..")
from generator.Generator import *
from generator import base_numbers
from models import get_model


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


if __name__ == "__main__":
    model = get_model()

    model.load_weights("model.h5", by_name=True)

    for i in range(100):
        print("Trial %s" % i)

        gen = Generator(base_numbers)
        print("start solved ?", gen.board.is_solved())

        gen.randomize(np.random.randint(31, 360))

        initial = gen.board.copy()

        print("shuffled solved ?", initial.is_solved())

        gen.reduce_via_logical(np.random.randint(31, 360))

        removed = gen.board.copy()

        print("removed solved ?", removed.is_solved())

        x_in = [a.value for x in removed.rows.values() for a in x]

        pred_gen = predict(x_in, model)

        print("Prediction solved ?", pred_gen.board.is_solved())

    x_new = [3, 1, 0, 0, 6, 0, 0, 0, 0,
             0, 0, 9, 0, 0, 0, 6, 2, 0,
             4, 0, 0, 0, 9, 8, 0, 0, 3,
             0, 0, 3, 9, 8, 0, 0, 4, 1,
             0, 7, 8, 1, 0, 0, 0, 5, 0,
             0, 0, 2, 6, 0, 5, 0, 0, 9,
             0, 0, 0, 0, 4, 6, 7, 9, 8,
             5, 6, 0, 0, 0, 9, 1, 0, 0,
             8, 0, 7, 0, 0, 3, 4, 0, 0]

    pred_gen = predict(x_new, model)

    print("Medium Prediction solved ?", pred_gen.board.is_solved())
    print(pred_gen.board)
