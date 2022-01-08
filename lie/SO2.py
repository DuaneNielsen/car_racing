import jax.numpy as np
import lie


class so2:
    def __init__(self, theta=0.0):
        self.theta = theta

    def as_matrix(self):
        return np.array([
            [0., -1],
            [1., 0]
        ]) * self.theta

    def __mul__(self, other):
        if np.isscalar(other):
            self.theta = self.theta * other
            return self

    def __add__(self, other):
        return so2(self.theta + other.theta)


class SO2:
    def __init__(self, theta):
        self.theta = theta

    def as_matrix(self):
        return np.array([
            [np.cos(self.theta), -np.sin(self.theta)],
            [np.sin(self.theta), np.cos(self.theta)]
        ])


def exp(so2_):
    if not isinstance(so2_, so2):
        raise lie.LieException('can only exponential map so2 -> SO2')
    return SO2(so2_.theta)


def log(SO2_):
    if not isinstance(SO2_, SO2):
        raise lie.LieException('can only log map S02 -> so2')
    return so2(SO2_.theta)