from matplotlib import pyplot as plt
import numpy as np
from math import sin, cos, radians
import icp

"""
Demo of ICP algo that aligns 2 lines
"""


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


x = np.linspace(0, 6, 6)
y = 0.5 * x ** 2 - 0.08 * x ** 3
target = np.stack([x, y])

fig = plt.figure()
ax = fig.subplots(1, 1)

source = np.matmul(se2(0.0, 0.1, radians(10.0)), homo(x, y))[0:2]

ax.plot(*unstack(source))

while icp.rms(source, target) > 0.001:
    R, t = icp.icp(source, target)
    source = np.matmul(R, source + t)

    ax.clear()
    ax.plot(*unstack(source))
    ax.plot(*unstack(target))
    print(icp.rms(source, target))
    plt.pause(1.0)

plt.show()
