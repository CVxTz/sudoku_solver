from generator.Generator import *
from generator import base_numbers
import numpy as np


def replace_with_zeros(x_arr, ratio=0.5):
    out = []
    for x in x_arr:
        if np.random.uniform(0, 1) < ratio:
            out.append(x)
        else:
            out.append(0)

    return out


def get_sample(random_replace=True):

    gen = Generator(base_numbers, shuffle_base=True)
    gen.randomize(np.random.randint(20, 1000))
    initial = gen.board.copy()
    x_out = [a.value for x in initial.rows.values() for a in x]

    gen.reduce_via_logical(np.random.randint(31, 360))

    removed = gen.board.copy()

    x_in = [a.value for x in removed.rows.values() for a in x]

    if random_replace:
        x_in = replace_with_zeros(x_in, ratio=0.6)

    return x_in, x_out


def gen(batch_size=32):
    while True:
        samples = [get_sample() for _ in range(batch_size)]
        X, Y = zip(*samples)
        X_pos = [list(range(81))] * batch_size

        yield [np.array(X), np.array(X_pos)], np.array(Y)[..., np.newaxis]
