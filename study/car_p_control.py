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
Naive Proportional control for car
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
fig, axes = plt.subplots(1, 2)
plt_steer, plt_spd = axes

origins = torch.tensor([
    [60., 60., 60., 60., 60.],
    [48., 48., 48., 48., 48.],
]).T
angles = torch.tensor(
    [radians(180. + 45.), radians(180. + 27.), radians(180.), radians(180. - 27.), radians(180. - 45.)])
steering_directions = ['hard_left', 'easy_left', 'straight', 'easy_right', 'hard_right']
steering_directions_a = [-1.0, -0.5, 0., 0.5, 1.0]
steering_speed = ['brake', 'pedal_off', 'pedal_on']


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


def max_beam(distance, end):
    i = torch.argmax(distance)
    return i, angles[i], distance[i], end[i], steering_directions[i], steering_directions_a[i]


class Speedo:
    def __init__(self):
        self.prev = None

    def speed(self, d):
        if self.prev is None:
            self.prev = d
            return np.zeros_like(d)
        else:
            spd = d - self.prev
            self.prev = d
            return spd


speedo = Speedo()
done = False

while True:
    obs, reward, done, info = env.step(a)
    sdf, sign = vision_utils.road_distance_field(obs, vision_utils.road_segment)
    sdf = sdf
    sdf = torch.from_numpy(sdf)
    end, distance = sphere_march(origins, vec_norm, sdf)
    argmax, steering_angle, steering_distance, steering_end, steering_direction, steering_a = max_beam(distance, end)
    a[0] = steering_a
    speed = speedo.speed(distance)
    a[1] = 0.3 if speed[2] > -2. else 0.0
    a[2] = 0.7 if speed[2] < -3. else 0.0


    plt_steer.clear()
    plt_steer.imshow(sdf.T)
    line = torch.stack((origins, end), dim=-1)
    for i in range(line.shape[0]):
        color='blue'
        if i == argmax:
            color = 'red'
        plt_steer.plot(line[i, 0], line[i, 1], color=color)

    plt_spd.clear()
    plt_spd.imshow(sdf.T)
    color = 'red' if speed[2] < -3. else 'blue'
    plt_spd.plot(line[2, 0], line[2, 1], color=color)
    plt.pause(0.01)
    env.render()
