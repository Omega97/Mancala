from random import random


def i_print(itr, n=-1):
    for i, x in enumerate(itr):
        if i==n:
            break
        print(x)
    print()


def mancala_gen(start, n_stones, board_size):
    """where to drop the n_stones stones grabbed from start"""
    i = start + 1
    while n_stones:
        if i == (board_size+1) * 2:
            i = 0
        if i != start:
            yield i
        n_stones -= 1
        i += 1


def step(x):
    if x > 0:
        return 1
    elif x < 0:
        return 0
    else:
        return .5


def adjust_values(v, k=1.):
    """ early values get closer to later values

    :param v: list of values
    :param k:
    k = 0  ->  no change
    k = 1  ->  values change visibly
    k = inf  ->  all values set to result
    :return:
    """
    if k <= 0:
        return v
    else:
        x = v[-1]
        out = [x]
        for i in reversed(range(len(v)-1)):
            x = v[i] + (x-v[i]) * 2**(-1/k)
            out = [x] + out
        return out


def choose(v):
    s = sum(v)
    if s <= 0:
        raise ValueError('sum must be > 0')

    r = random() * s

    for i in range(len(v)):
        r -= v[i]
        if r < 0:
            return i
