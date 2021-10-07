from matplotlib import pyplot as plt
import numpy as np
from numpy.linalg import svd
from math import sin, cos, radians


def unstack(line):
    return line[0], line[1]


def homo(x, y):
    h = np.ones_like(x)
    return np.stack([x, y, h])


def se2(x, y, theta):
    return np.array([
        [cos(theta), sin(theta), x],
        [-sin(theta), cos(theta), y],
        [0.0, 0, 1]
    ])


def rms(source, target):
    return np.sum((source - target) ** 2) / source.shape[1]


def icp(source, target):
    """

    :param source: set of source points
    :param target: set of target points
    :return:
    """
    source_center = source.mean(axis=1, keepdims=True)
    target_center = target.mean(axis=1, keepdims=True)
    t = - target_center + source_center
    target = target + t
    corr = np.matmul(source, target.T)
    u, d, v = svd(corr)
    R = np.matmul(u, v.T)
    target = np.matmul(R, target)
    return target, R, t


x = np.linspace(0, 6, 6)
y = 0.5 * x ** 2 - 0.08 * x ** 3
source = np.stack([x, y])

fig = plt.figure()
ax = fig.subplots(1, 1)

target = np.matmul(se2(0.0, 0.1, radians(10.0)), homo(x, y))[0:2]

ax.plot(*unstack(target))

while rms(source, target) > 0.001:
    target, R, t = icp(source, target)

    ax.clear()
    ax.plot(*unstack(source))
    ax.plot(*unstack(target))
    print(rms(source, target))
    plt.pause(1.0)

plt.show()
