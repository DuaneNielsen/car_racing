import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon, Circle
import geometry as geo
import icp


def draw_frame(ax, frame, centroid, centroid_r, color):
    ax.add_patch(Polygon(geo.transform_points(frame.M, frame.vertices).T, color=color, fill=False))
    centroid_w = geo.transform_points(frame.M, centroid)
    ax.add_patch(Circle((centroid_w[0], centroid_w[1]), radius=3, color=color, fill=False))
    x_ax = np.matmul(centroid_r, np.array([3, 0]).reshape(2, 1))
    x_ax = np.concatenate((centroid_w, centroid_w + x_ax), axis=1)
    ax.plot(x_ax[0], x_ax[1], color=color)


def draw_vector(ax, p, v):
    line = np.concatenate([p, p + v], axis=1)
    ax.plot(line[0], line[1])


if __name__ == '__main__':

    fig = plt.figure()
    ax = fig.subplots(1, 1)
    ax.set_ylim(-100, 100)
    ax.set_xlim(-10, 100)
    ax.set_aspect('equal')

    h, w = 30, 40

    f0 = geo.Scan(h, w, x=5., y=-15., theta=np.radians(10.0))
    f0.centroid = np.array([30, 15]).reshape(2, 1)
    f0.centroid_R = geo.R(np.radians(20.0))

    f1 = geo.Scan(h, w, x=-5., y=10., theta=np.radians(-20.0))
    f1.centroid = np.array([20, 20]).reshape(2, 1)
    f1.centroid_R = geo.R(np.radians(-20.0))

    f1_aligned = geo.Scan(h, w, x=-5., y=10., theta=np.radians(-20.0))
    f1_aligned.centroid = np.array([20, 20]).reshape(2, 1)
    f1_aligned.centroid_R = geo.R(np.radians(-20.0))

    f0_centroid_w = geo.transform_points(f0.M, f0.centroid)
    f1_centroid_w = geo.transform_points(f1.M, f1.centroid)

    R = np.matmul(f0.centroid_R, f1.centroid_R.T)
    t = f0_centroid_w - f1_centroid_w

    # rotate
    M = geo.R_around_point_Rt(R, f1_centroid_w)
    M[0:2, 2:] += t
    f1_aligned.M = np.matmul(M, f1_aligned.M)
    f1_aligned.centroid_R = np.matmul(R, f1_aligned.centroid_R)

    ax.clear()
    draw_frame(ax, f0, f0.centroid, f0.centroid_R, 'blue')
    draw_frame(ax, f1, f1.centroid, f1.centroid_R, 'green')
    draw_frame(ax, f1_aligned, f1_aligned.centroid, f1_aligned.centroid_R, 'red')
    plt.show()