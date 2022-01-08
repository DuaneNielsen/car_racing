import numpy as np
from matplotlib import pyplot as plt
import geometry as geo


def R_around_point(theta, x=0.0, y=0.0):
    return np.array([
        [np.cos(theta), -np.sin(theta), x - np.cos(theta) * x + np.sin(theta) * y],
        [np.sin(theta), np.cos(theta), y - np.sin(theta) * x + np.cos(theta) * y],
        [0., 0., 1.]
    ])


def R_around_point_from_R(R, x, y):
    return np.array([
        [R[0, 0], R[0, 1], x - R[0, 0] * x - R[0, 1] * y],
        [R[1, 0], R[1, 1], y - R[1, 0] * x - R[1, 1] * y],
        [0., 0., 1]
    ])


fig = plt.figure()
ax = fig.subplots(1)

P_ = np.array([
    [0, 1]
]).T

theta, x, y = -.5, 0.5, 0.

for _ in range(200):
    ax.clear()
    ax.set_aspect('equal')
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.autoscale(False)
    R = geo.R(theta)
    P = geo.transform_points(geo.R_around_point_Rt(R, np.array([x, y]).reshape(2, 1)), P_)
    ax.scatter(P[0], P[1])
    ax.scatter(x, y)
    theta += np.radians(3.0)
    plt.pause(0.1)
