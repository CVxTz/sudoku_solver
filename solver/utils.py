import numpy as np

from generator import base_numbers
from generator.Generator import *


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
    x_out = [[a.value for a in x] for x in initial.rows.values()]

    gen.reduce_via_logical(np.random.randint(31, 360))

    removed = gen.board.copy()

    x_in = [[a.value for a in x] for x in removed.rows.values()]

    if random_replace:
        x_in = replace_with_zeros(x_in, ratio=np.random.uniform(0.6, 0.9))

    return x_in, x_out


def binarize_along_last_axis(arr, n_classes=10):
    out = np.zeros(arr.shape + (n_classes,))
    for i in range(n_classes):
        out[..., i] = (arr == i).astype(np.float)

    return out


def gen(batch_size=64):
    while True:
        samples = [get_sample(random_replace=False) for _ in range(batch_size)]
        X, Y = zip(*samples)
        X = np.array(X)
        Y = np.array(Y)

        X = binarize_along_last_axis(X, n_classes=10)
        Y = binarize_along_last_axis(Y, n_classes=10)

        yield X, Y
