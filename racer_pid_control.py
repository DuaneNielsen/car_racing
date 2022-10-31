import gym
import vision_utils
from pyglet.window import key
import numpy as np
from matplotlib import pyplot as plt
import env
import torch
from math import radians
from torch.nn.functional import interpolate
from collections import deque
from torch import multiprocessing as mp
from tqdm import tqdm
import time
from gym.wrappers import TimeLimit

"""
PID control of car from visual input
"""

a = np.array([0, 0], dtype=np.int32)


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


def angle_to_norm(angle):
    x = torch.cos(angle)
    y = torch.sin(angle)
    return torch.stack((x, y), dim=1)


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
    x0, y0 = torch.floor(x).clamp(0, W - 1).long(), torch.floor(y).clamp(0, W - 1).long()
    x1, y1 = torch.ceil(x).clamp(0, H - 1).long(), torch.ceil(y).clamp(0, H - 1).long()
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
        distance = interp_dist_from_sdf_grid(origin, sdf)
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


class Evaluator():
    def __init__(self, num_runs=1, render=False, log=False, demo=False, fps=1/60):
        self.num_runs = num_runs
        self.render, self.log, self.demo, self.fps = render, log, demo, fps

        render_mode = 'human' if render else 'console'
        self.env = TimeLimit(gym.make('racer-v0', render_mode=render_mode), max_episode_steps=500)

    def __call__(self, batch_params, demo=False):
        return self.run_batch(batch_params)

    def run_batch(self, batch_params):
        results = []
        for params in batch_params:
            results += [self.episode(params)]
        return np.array(results)

    def episode(self, params):
        """
        params: [P_throttle, I_throttle, D_throttle, P_steering, I_steering, D_steering, max_speed]
        """
        P_t, I_t, D_t = params[0], params[1], params[2]
        P_s, I_s, D_s = params[3], params[4], params[5]
        max_speed = params[6]

        if self.render:
            self.env.render()

        reward_total = 0

        for _ in range(self.num_runs):

            obs, done = self.env.reset(), False
            steering_directions_a = torch.linspace(-1., 1., len(obs))
            a = self.env.action_space.sample()

            integral = deque([20.], maxlen=5)
            integral_steering = deque([0.], maxlen=5)

            while not done:
                obs, reward, done, info = self.env.step(a)
                obs = torch.from_numpy(obs)
                reward_total += reward
                speed = info['car_speed'] / 40.

                # get the longest ray, this is the direction we want to go
                i = torch.argmax(obs)
                idxs = obs == obs[i]
                i += torch.div(idxs.sum(), 2, rounding_mode='trunc')  # if there is a patch of sensors at same length, select the center one
                dist = obs[i].item() / 20.

                # PID controller - steering
                integral_steering_D = steering_directions_a[i] - integral_steering[-1]
                integral_steering.append(steering_directions_a[i])
                steer = steering_directions_a[i] * P_s + sum(integral_steering) / len(
                    integral_steering) * I_s + integral_steering_D * D_s

                # PID controller - speed
                derivative = dist - integral[-1]
                integral.append(dist)
                throttle = dist * P_t + sum(integral) / len(integral) * I_t + derivative * D_t
                # print(throttle, speed)

                a[0] = 0 if abs(throttle) < 0.5 else 1 if throttle > 0.5 else 2
                a[0] = 2 if speed > max_speed else a[0]  # slow down if max speed is exceeded
                a[1] = 0 if abs(steer) < 0.5 else 1 if steer < -0.5 else 2

                if self.log:
                    print(
                        f'reward: {reward_total} speed: {speed:.2f}, throttle: {a[0]}, steering: {a[1]}')

                if self.render:
                    self.env.render(mode='human', reward=reward_total)
                    time.sleep(self.fps)

        return reward_total


if __name__ == '__main__':
    eval = Evaluator(render=True, log=True, num_runs=1)

    """
    t -> throttle, s -> steering
    params = [P_t, I_t, D_t, P_s, I_s, D_s, max_speed]
    max speed will be multiplied by 100. ie: 0.54 = max speed of 54
    """

    params = [
        [0.0640, 0.5743, 0.6588, -0.3052, 1.9942, -0.1781, 0.2249],
        [-0.0279, 1.4524, 0.8553, 0.7620, 0.8637, 0.5132, 0.2455],
        [0.3194, 0.9998, 0.6312, -0.3236, 2.0617, -0.3036, 0.2185],
        [0.3197, 1.0159, 0.5003, -0.3049, 2.0153, -0.3526, 0.2076],
        [0.3488, 1.0490, 0.4762, -0.2328, 1.9820, -0.4044, 0.2128]
    ]

    for p in params:
        reward = eval.episode(p)
        print(reward)
