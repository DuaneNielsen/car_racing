import torch
import numpy as np
from matplotlib import pyplot as plt
import dataloaders
from math import sin, cos, radians


def ray_march(origin, vec, sdf, initial_polarity):
    x, y = origin[:, 0].long(), origin[:, 1].long()
    x, y = x.clamp(0, 219), y.clamp(0, 219)
    distance = sdf[x, y] * initial_polarity
    polarity_at = torch.sign(sdf[x, y])
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
           [108., 108., 108., 108., 108.],
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