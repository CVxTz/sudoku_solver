from generator.Generator import *
from generator import base_numbers
import numpy as np


def get_sample():

    gen = Generator(base_numbers)
    gen.randomize(100)
    initial = gen.board.copy()
    x_out = [a.value for x in initial.rows.values() for a in x]

    gen.reduce_via_logical(81)

    removed = gen.board.copy()

    x_in = [a.value for x in removed.rows.values() for a in x]
    return x_in, x_out


def gen(batch_size=32):
    while True:
        samples = [get_sample() for _ in range(batch_size)]
        X, Y = zip(*samples)
        X_pos = [list(range(81))] * batch_size

        yield [np.array(X), np.array(X_pos)], np.array(Y)[..., np.newaxis]
