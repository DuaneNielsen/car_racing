import torch
from torch.nn.functional import interpolate
import numpy as np
from matplotlib import pyplot as plt
import dataloaders
from math import sin, cos, radians
from torch import stack


def ray_march(origin, vec, sdf, initial_polarity):
    N, _ = vec.shape
    x, y = origin[:, 0], origin[:, 1]

    # interpolate 4 nearest grid cells
    x0, y0 = torch.floor(x).clamp(0, 219).long(), torch.floor(y).clamp(0, 219).long()
    x1, y1 = torch.ceil(x).clamp(0, 219).long(), torch.ceil(y).clamp(0, 219).long()
    x = torch.stack([x0, x0, x1, x1], dim=-1).reshape(N, 2, 2)
    y = torch.stack([y0, y1, y0, y1], dim=-1).reshape(N, 2, 2)
    grid = sdf[x, y]
    interp_distance = interpolate(grid.unsqueeze(1), size=(1, 1), mode='bilinear', align_corners=True).squeeze()
    polarity_at = torch.sign(interp_distance)
    distance = interp_distance * initial_polarity
    end_trace = (initial_polarity * polarity_at) > 0.
    return origin + vec * distance.unsqueeze(-1) * end_trace.unsqueeze(-1)


def angle_to_norm(angle):
    x = torch.cos(angle)
    y = torch.sin(angle)
    return torch.stack((x, y), dim=1)


def get_polarity(origin, sdf):
    x, y = origin[:, 0].long(), origin[:, 1].long()
    return torch.sign(sdf[x, y])


if __name__ == '__main__':
       loader = dataloaders.NPZLoader('../data/road_sdfs/')
       fig, ax = plt.subplots(1)
       fig.suptitle('Sphere tracing a Signed Distance Function')

       """
       Number of rays, Number of spatial dimensions, number of steps of marching 
       N, 2, S
       """

       origins = torch.tensor([
           [107.5, 107.5, 107.5, 107.5, 107.5],
           [219., 219., 219., 219., 219.],
       ]).T

       angles = torch.tensor(
           [radians(270. - 45.), radians(270. - 27.), radians(270.), radians(270. + 27.), radians(270. + 45.)])
       vec_norm = angle_to_norm(angles)

       for i in range(30):
           sdf, label = loader[i]
           polarity = get_polarity(origins, sdf)

           steps = [origins]

           for _ in range(6):
               step = ray_march(steps[-1], vec_norm, sdf, polarity)
               steps += [step]
               rays = torch.stack(steps, dim=-1)

               ax.clear()
               ax.imshow(sdf.T)
               ax.set_xlim(0, 219.)
               ax.set_ylim(0, 219.)
               for ray in rays:
                   ax.plot(ray[0], ray[1])
               plt.pause(0.5)

       plt.show()