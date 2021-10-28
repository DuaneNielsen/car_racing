import torch
import numpy as np
from matplotlib import pyplot as plt

"""
Thanks to....

A Volumetric Method for Building Complex Models from Range Images

Brian Curless and Marc Levoy

https://graphics.stanford.edu/papers/volrange/volrange.pdf
"""

if __name__ == '__main__':

    i = 1
    poses = torch.from_numpy(np.load(f'data/ep{i}_pose.npy')).float()
    sdfs = torch.from_numpy(np.load(f'data/ep{i}_sdf.npy')).float().permute(0, 2, 1)
    states = torch.from_numpy(np.load(f'data/ep{i}_state.npy'))

    fig = plt.figure()
    sdf_plot = fig.subplots(1, 1)

    N, h, w = sdfs.shape
    batch_size = 8
    world_h, world_w = (800, 400)

    world_map = torch.zeros((world_h, world_w))
    world_map_N = torch.zeros((world_h, world_w))

    offset = torch.tensor([400, 300]).reshape(2, 1)

    def make_grid(h, w, homo=False):
        axis = []
        axis += [*torch.meshgrid(torch.arange(h), torch.arange(w))]
        if homo:
            axis += [torch.ones_like(axis[0])]
        return torch.stack(axis, dim=2)

    # transform to world space
    model_grid = make_grid(h, w, homo=True).float().reshape(h * w, 3).T
    X = torch.matmul(poses.float(), model_grid.float())[:, 0:2]
    X = X.reshape(X.shape[0], X.shape[1], h, w).permute(0, 2, 3, 1)
    X += torch.tensor([400., 300.]).reshape(1, 1, 1, 2)
    X = torch.round(X).long()

    # combine the scans into truncated SDF
    for index, sdf in zip(X, sdfs):
        index_h, index_w = index.flatten(0, 1)[:, 0], index.flatten(0, 1)[:, 1]
        N = world_map_N[index_h, index_w]
        world_map[index_h, index_w] = (world_map[index_h, index_w] * N + sdf.flatten(0, 1).clip(-4.0, +4.0))/ (N + 1.0)
        world_map_N[index_h, index_w] += 1.0

        sdf_plot.clear()
        sdf_plot.imshow(world_map.T, cmap='cool')
        plt.pause(0.05)

    plt.show()