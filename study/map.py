import numpy as np
import geometry as geo
from matplotlib import pyplot as plt

if __name__ == '__main__':
    ep = 1
    pose = np.load(f'data/ep{0}_pose.npy')
    sdf = np.load(f'data/ep{0}_sdf.npy')
    state = np.load(f'data/ep{0}_state.npy')

    world = np.zeros((400, 400))
    world_origin = np.array([200., 200.]).reshape(2, 1)

    T, H, W = sdf.shape

    for t in range(T):
        x_i, y_i = np.meshgrid(np.arange(H), np.arange(W))
        x_i, y_i = x_i.flatten(), y_i.flatten()
        p = np.stack((x_i, y_i))
        p = geo.transform_points(pose[t], p) + world_origin
        p = np.round(p).astype(int)
        x_i_w, y_i_w = p[0], p[1]
        world[x_i_w, y_i_w] = sdf[t, x_i, y_i]

        plt.imshow(world, cmap='cool')
        plt.pause(0.1)