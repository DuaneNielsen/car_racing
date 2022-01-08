import torch
from torch import nn
from torch.nn.functional import mse_loss
from torch.optim import Adam
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


def draw_dataset(X, sdfs):
    world = torch.zeros((world_x, world_y)).float()
    for x, sdf in zip(X, sdfs):
        x = torch.round(x).long()
        world[x[0], x[1]] = sdf[model_grid[0], model_grid[1]]
    plt.imshow(world)
    plt.show()


if __name__ == '__main__':

    i = 1
    poses = torch.from_numpy(np.load(f'data/ep{i}_pose.npy'))
    sdfs = torch.from_numpy(np.load(f'data/ep{i}_sdf.npy')).float()
    states = torch.from_numpy(np.load(f'data/ep{i}_state.npy'))

    N, h, w = sdfs.shape
    world_x, world_y = (800, 400)
    world_grid_xy = torch.stack(torch.meshgrid(torch.arange(world_x), torch.arange(world_y))).reshape(2, world_x * world_y)

    def make_grid(h, w, homo=False):
        axis = []
        axis += [*torch.meshgrid(torch.arange(h), torch.arange(w))]
        if homo:
            axis += [torch.ones_like(axis[0])]
        return torch.stack(axis, dim=2)

    model_grid = make_grid(h, w)

    offset = torch.tensor([400, 300]).reshape(2, 1)
    # X = torch.matmul(poses.float(), model_grid.T.float())[:, 0:2] + offset

    #ds = TensorDataset(X, sdfs)
    gt = sdfs[0:1].unsqueeze(-1).clip(-1.0, 4.0)
    grid = make_grid(h, w, homo=False).float().unsqueeze(0)
    ds = TensorDataset(grid, gt)
    dl = DataLoader(ds, batch_size=1)

    fig = plt.figure()
    axes = fig.subplots(1, 2)
    sdf_plot, gt_plot = axes

    layersize = 512

    def block(out_params, in_params=None, batchnorm=False):
        in_params = out_params if in_params is None else in_params
        if batchnorm:
            return nn.Sequential(nn.Linear(in_params, out_params), nn.BatchNorm1d(out_params), nn.ELU(inplace=True))
        else:
            return nn.Sequential(nn.Linear(in_params, out_params), nn.ELU(inplace=True))

    sdf = nn.Sequential(block(in_params=2, out_params=layersize, batchnorm=False),
                        block(layersize),
                        block(layersize),
                        nn.Linear(layersize, 1, bias=False)).cuda()
    optim = Adam(sdf.parameters(), lr=1e-3)

    for epoch in range(3000):
        for x, sd in dl:
            x, sd = x.cuda(), sd.cuda()
            optim.zero_grad()
            d = sdf(x)
            loss = mse_loss(d, sd)
            loss.backward()
            optim.step()
            print(loss)

            if epoch % 10 == 0:
                gt_plot.clear()
                sdf_plot.clear()
                gt_plot.imshow(sd.squeeze().detach().cpu(), cmap='cool')
                sdf_plot.imshow(d.squeeze().detach().cpu(), cmap='cool')
                plt.pause(0.05)

plt.show()

