import torch
from torch import nn
from torch.nn.functional import mse_loss, elu
from torch.optim import Adam
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


# def draw_trajectory():
#     fig, ax = plt.subplots()
#     model_grid = make_grid(h, w, homo=True).float().reshape(h * w, 3).T
#     X = torch.matmul(poses.float(), model_grid.float())[:, 0:2] + offset
#     X = X.reshape(X.shape[0], X.shape[1], h, w).permute(0, 2, 3, 1)
#     gt_image = torch.zeros((world_h, world_w))
#
#     for index, scan in zip(X, sdfs):
#         gt_image[index.long().flatten(0, 1)[:, 0], index.long().flatten(0, 1)[:, 1]] = scan.flatten(0, 1).squeeze()
#         ax.clear()
#         ax.imshow(gt_image.detach().cpu(), cmap='cool')
#         plt.pause(0.05)


if __name__ == '__main__':

    with torch.no_grad():
        i = 1
        poses = torch.from_numpy(np.load(f'data/ep{i}_pose.npy')).float()
        sdfs = torch.from_numpy(np.load(f'data/ep{i}_sdf.npy')).float().permute(0, 2, 1)
        states = torch.from_numpy(np.load(f'data/ep{i}_state.npy'))

    N, h, w = sdfs.shape
    batch_size = 8
    world_h, world_w = (800, 400)
    world_h, world_w = (120, 120)

    def make_grid(h, w, homo=False):
        axis = []
        axis += [*torch.meshgrid(torch.arange(h), torch.arange(w))]
        if homo:
            axis += [torch.ones_like(axis[0])]
        return torch.stack(axis, dim=2)

    world_grid = make_grid(world_h, world_w).float().cuda()
    model_grid = make_grid(h, w, homo=True).float().reshape(h * w, 3).T
    offset = torch.tensor([400, 300]).reshape(2, 1)

    # transform to world space
    X = torch.matmul(poses.float(), model_grid.float())[:, 0:2]
    X = X.reshape(X.shape[0], X.shape[1], h, w).permute(0, 2, 3, 1)

    X, sdfs= X[0:1], sdfs[0:1]

    # normalize to range 0 .. 1
    # just be aware this is DATA DEPENDENT, SO NEW DATA WILL NOT MAP TO SAME CO-ORDINATES
    X_min = X.amin(0).amin(0).amin(0).reshape(1, 1, 1, 2)
    X_max = X.amax(0).amax(0).amax(0).reshape(1, 1, 1, 2)
    X = (X - X_min) / X_max
    sdfs = (sdfs - sdfs.min()) / sdfs.max()

    gt = sdfs.unsqueeze(-1)
    ds = TensorDataset(X, gt)
    dl = DataLoader(ds, batch_size=1, shuffle=True)

    fig = plt.figure()
    axes = fig.subplots(1, 2)
    sdf_plot, gt_plot = axes

    layersize = 512

    class Block(nn.Module):
        def __init__(self, out_params, in_params=None):
            super().__init__()
            in_params = out_params if in_params is None else in_params
            self.linear = nn.Linear(in_params, out_params)
            self.bn = nn.BatchNorm1d(out_params)

        def forward(self, x):
            shape = x.shape
            x = self.linear(x)
            # x_flat = self.bn(x.flatten(0, -2))
            # x = x_flat.reshape(*shape[:-1], x_flat.shape[-1])
            return elu(x)

    sdf = nn.Sequential(Block(in_params=2, out_params=layersize),
                        Block(layersize),
                        Block(layersize),
                        Block(layersize),
                        Block(layersize),
                        nn.Linear(layersize, 1, bias=False)).cuda()
    optim = Adam(sdf.parameters(), lr=1e-4)

    for epoch in range(30000):
        for x, sd in dl:
            x, sd = x.cuda(), sd.cuda()
            optim.zero_grad()
            d = sdf(x)
            loss = mse_loss(d, sd)
            loss.backward()
            optim.step()
            print(loss)

            if epoch % 100 == 0:
                with torch.no_grad():
                    image = sdf(world_grid).squeeze()
                    sdf_plot.imshow(image.squeeze().detach().cpu(), cmap='cool')
                    gt_image = torch.zeros((world_h, world_w), device='cuda')
                    for index, scan in zip(x, sd):
                        index = index * X_max.reshape(1, 1, 2).cuda()
                        gt_image[index.long().flatten(0, 1)[:, 0], index.long().flatten(0, 1)[:, 1]] = scan.flatten(0, 1).squeeze()
                    gt_plot.imshow(gt_image.detach().cpu(), cmap='cool')
                    plt.pause(0.05)


plt.show()