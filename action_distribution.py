import numpy as np
from copy import copy

try:
    from utils import choose
except ImportError:
    from .utils import choose


class ActionDistribution:

    def __init__(self, v):
        self.v = np.array(v).astype(np.float32)

    def __getitem__(self, item):
        return self.v[item]

    def __setitem__(self, key, value):
        self.v[key] = value

    def __len__(self):
        return len(self.v)

    def __repr__(self):
        v = self.v / self.norm()
        return '[' + '  '.join(f'{i:.3f}' for i in v) + ']'

    def __mul__(self, other):
        if type(other) in [int, float]:
            return ActionDistribution(self.v * other)
        else:
            assert len(self) == len(other)
            return ActionDistribution([self[i] * other[i] for i in range(len(self))])

    def __add__(self, other):
        if type(other) == type(self):
            return ActionDistribution(self.v + other.v)
        else:
            raise ValueError

    def __iter__(self):
        return iter(self.v)

    def __copy__(self):
        return ActionDistribution(copy(self.v))

    def norm(self):
        return sum(self.v)

    def normalize(self):
        self.v /= self.norm()
        return self

    def quick_choose(self) -> int:
        """return an index of a random element using self.v as weights"""
        return choose(self.v)

    def choose_move(self):
        """return a one-hot AD of a random element using self.v as weights"""
        index = self.quick_choose()
        return one_hot(index, len(self))

    def argmax(self):
        return int(np.argmax(self.v))

    def keep_max(self):
        """all elements become 0 except max becomes 1"""
        i = self.argmax()
        return one_hot(i, len(self))

    def focus(self, k):
        """
        k = 0 -> don't change
        k = 1 -> keep max
        :param k: float (0., 1.)
        """
        if k <= 0:
            return self
        elif k >= 1:
            return self.keep_max()
        else:
            x = max(self.v) * k
            new_v = [(i-x) if (i-x) > 0 else 0. for i in self.v]
            return ActionDistribution(new_v)

    def get_non_zero(self):
        return tuple(i for i in range(len(self.v)) if self.v[i])

    def get_move_index(self):
        v = self.get_non_zero()
        assert len(v) == 1, 'move should be a one-hot vector'
        return v[0]

    def complementary(self):
        return ActionDistribution([0 if i else 1 for i in self])


def one_hot(i, size):
    return ActionDistribution([1 if j == i else 0 for j in range(size)])
