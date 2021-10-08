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

    :param source: set of source points, D, N
    :param target: corresponding set of target points, D, N
    :return: corrected source points, rotation matrix, translation
    """
    source_center = source.mean(axis=1, keepdims=True)
    target_center = target.mean(axis=1, keepdims=True)
    t = - target_center + source_center
    source = target + t
    corr = np.matmul(target, source.T)
    u, d, v = svd(corr)
    R = np.matmul(u, v.T)
    source = np.matmul(R, source)
    return source, R, t


x = np.linspace(0, 6, 6)
y = 0.5 * x ** 2 - 0.08 * x ** 3
target = np.stack([x, y])

fig = plt.figure()
ax = fig.subplots(1, 1)

source = np.matmul(se2(0.0, 0.1, radians(10.0)), homo(x, y))[0:2]

ax.plot(*unstack(source))

while rms(source, target) > 0.001:
    source, R, t = icp(source, target)

    ax.clear()
    ax.plot(*unstack(source))
    ax.plot(*unstack(target))
    print(rms(source, target))
    plt.pause(1.0)

plt.show()
