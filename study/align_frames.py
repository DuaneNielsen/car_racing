import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon, Circle
import geometry as geo
import icp
import copy


def draw_frame_relative(ax, frame, color):
    ax.add_patch(Polygon(frame.vertices.T, color=color, fill=False))
    ax.add_patch(Circle(np.split(frame.centroid, 2), radius=1, color=color, fill=False))
    ax.scatter(frame.kp[0], frame.kp[1], color=color)


def draw_frame(ax, frame, color):
    ax.add_patch(Polygon(frame.vertices_w.T, color=color, fill=False))
    ax.add_patch(Circle(np.split(frame.centroid_w, 2), radius=3, color=color, fill=False))
    ax.scatter(*np.split(frame.kp_w, 2), color=color)


def draw_vector(ax, p, v):
    line = np.concatenate([p, p + v], axis=1)
    ax.plot(line[0], line[1])


if __name__ == '__main__':

    fig = plt.figure()
    axes = fig.subplots(1, 3)
    world_plt, f0_plt, f1_plt = axes
    world_plt.set_ylim(-100, 100)
    world_plt.set_xlim(-10, 100)
    [ax.set_aspect('equal') for ax in axes]

    h, w = 30, 40

    kp = np.array([
        [5.0, 13.0],
        [12.0, 10.0],
        [8.0, 19.0],
        [19.0, 7.0],
    ]).T

    f0 = geo.Scan(h, w, x=5., y=-15., theta=np.radians(10.0))
    f0.kp = kp

    for t in range(300):

        f1 = geo.Scan(h, w, x=10., y=-15., theta=np.radians(-10.0))
        kp_w = geo.transform_points(f0.M, f0.kp)
        f1.kp = geo.transform_points(f1.inv_M, kp_w)
        f1.R = np.matmul(geo.R(np.radians(-t)), f1.R)
        f1.t += np.array([10 * np.sin(np.radians(t)), 10]).reshape(2, 1)

        f1_aligned = copy.copy(f1)

        M = icp.icp_homo(f1_aligned.kp_w, f0.kp_w)
        f1_aligned.M = np.matmul(M, f1_aligned.M)

        [ax.clear() for ax in axes]
        draw_frame(world_plt, f0, 'blue')
        draw_frame_relative(f0_plt, f0, 'blue')

        draw_frame(world_plt, f1, 'green')
        draw_frame_relative(f1_plt, f1, 'green')

        draw_frame(world_plt, f1_aligned, 'red')
        plt.pause(0.05)

    plt.show()