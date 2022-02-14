import gym
import vision_utils
from pyglet.window import key
import numpy as np
from matplotlib import pyplot as plt
import env
import torch
from math import radians
from torch.nn.functional import interpolate

"""
Drive a car around the track and generate a dataset
"""

a = np.array([0.0, 0.0, 0.0])


def key_press(k, mod):
    global restart
    if k == 0xFF0D:
        restart = True
    if k == key.LEFT:
        a[0] = -1.0
    if k == key.RIGHT:
        a[0] = +1.0
    if k == key.UP:
        a[1] = +1.0
    if k == key.DOWN:
        a[2] = +0.8  # set 1.0 for wheels to block to zero rotation


def key_release(k, mod):
    if k == key.LEFT and a[0] == -1.0:
        a[0] = 0
    if k == key.RIGHT and a[0] == +1.0:
        a[0] = 0
    if k == key.UP:
        a[1] = 0
    if k == key.DOWN:
        a[2] = 0


env = gym.make('CarRacing-v1')
obs, done = env.reset(), False
action = env.action_space.sample()
env.render()
env.viewer.window.on_key_press = key_press
env.viewer.window.on_key_release = key_release
fig, ax = plt.subplots()

origins = torch.tensor([
    [60., 60., 60., 60., 60.],
    [48., 48., 48., 48., 48.],
]).T
angles = torch.tensor(
    [radians(180. + 45.), radians(180. + 27.), radians(180.), radians(180. - 27.), radians(180. - 45.)])


def angle_to_norm(angle):
    x = torch.cos(angle)
    y = torch.sin(angle)
    return torch.stack((x, y), dim=1)


vec_norm = angle_to_norm(angles)


def get_polarity(origin, sdf):
    x, y = origin[:, 0].long(), origin[:, 1].long()
    return torch.sign(sdf[x, y])


def interp_dist_from_sdf_grid(pos, sdf):
    """
    samples the 4 discrete grid cells surrounding a co-ordinate and interpolates
    to find the distance to the nearest surface
    N - the number of rays
    pos: N, 2 -> the position of each co-ordinate (x.y)
    """
    N, _ = pos.shape
    H, W = sdf.shape

    # interpolate 4 nearest grid cells
    x, y = pos[:, 0], pos[:, 1]
    x0, y0 = torch.floor(x).clamp(0, W-1).long(), torch.floor(y).clamp(0, W-1).long()
    x1, y1 = torch.ceil(x).clamp(0, H-1).long(), torch.ceil(y).clamp(0, H-1).long()
    x = torch.stack([x0, x0, x1, x1], dim=-1).reshape(N, 2, 2)
    y = torch.stack([y0, y1, y0, y1], dim=-1).reshape(N, 2, 2)
    grid = sdf[x, y]
    return interpolate(grid.unsqueeze(1), size=(1, 1), mode='bilinear', align_corners=True).squeeze()


def sphere_march(origin, vec, sdf, iterations=6):
    """
    sphere-traces N rays that terminate when the sdf == 0 (or num of iterations is exceeded)
    makes no assumption about our position being inside or outside of the surface
    origin: N, 2 -> the starting co-ordinate of N rays in sdf space
    vec: N, 2 -> a normal vector that points in the direction to trace
    sdf: H, W -> 2D discrete SDF (each grid cell contains the distance to the nearest surface)
    iterations: number of iterations to perform
    """
    with torch.no_grad():
        distance = interp_dist_from_sdf_grid(origins, sdf)
        total_distance = torch.zeros_like(distance)
        initial_polarity = torch.sign(distance)
        pos = origin.clone()

        for _ in range(iterations):
            distance = interp_dist_from_sdf_grid(pos, sdf)
            polarity_at = torch.sign(distance)
            distance = distance * initial_polarity
            end_trace = (initial_polarity * polarity_at) > 0.
            pos = pos + vec * distance.unsqueeze(-1) * end_trace.unsqueeze(-1)
            total_distance += distance
        return pos, total_distance * initial_polarity


while not done:
    obs, reward, done, info = env.step(a)
    sdf, sign = vision_utils.road_distance_field(obs, vision_utils.road_segment)
    sdf = sdf
    ax.clear()
    ax.imshow(sdf.T)
    sdf = torch.from_numpy(sdf)
    end, distance = sphere_march(origins, vec_norm, sdf)
    print(distance)
    line = torch.stack((origins, end), dim=-1)
    for i in range(line.shape[0]):
        ax.plot(line[i, 0], line[i, 1])
    plt.pause(0.01)
    env.render()
