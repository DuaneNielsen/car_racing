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

"""
PID control of car from visual input
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
    def __init__(self, plot=False, render=False, log=False, demo=False):
        self.plot, self.render, self.log, self.demo = plot, render, log, demo
        # self.env.viewer.window.on_key_press = key_press
        # self.env.viewer.window.on_key_release = key_release
        if plot:
            self.fig, self.axes = plt.subplots()
        else:
            self.fig, self.axes = None, None

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

        self.env = gym.make('CarRacing-v1')
        obs, done = self.env.reset(), False
        a = self.env.action_space.sample()

        if self.render:
            self.env.render()

        num_beams = 13
        origins = torch.tensor([
            [60.] * num_beams,
            [48.] * num_beams,
        ]).T
        angles = torch.tensor(
            [radians(180. + theta) for theta in torch.linspace(45., -45, num_beams)]
        )
        steering_directions_a = torch.linspace(-1., 1., num_beams)
        vec_norm = angle_to_norm(angles)

        integral = deque([51.], maxlen=5)
        integral_steering = deque([0.], maxlen=5)

        reward_total = 0

        while not done or self.demo:
            obs, reward, done, info = self.env.step(a)
            reward_total += reward
            speed = info['speed'].length / 100.

            # get the sdf
            sdf, sign = vision_utils.road_distance_field(obs, vision_utils.road_segment)
            sdf = torch.from_numpy(sdf)

            # cast rays using sphere marching
            end, distance = sphere_march(origins, vec_norm, sdf)

            # get the longest ray, this is the direction we want to go
            i = torch.argmax(distance)
            d = distance[i].item() / 55.
            dist = d if d > 0.1 else 0.1

            # PID controller - steering
            integral_steering_D = steering_directions_a[i] - integral_steering[-1]
            integral_steering.append(steering_directions_a[i])
            a[0] = steering_directions_a[i] * P_s + sum(integral_steering) / len(
                integral_steering) * I_s + integral_steering_D * D_s

            # PID controller - speed
            derivative = dist - integral[-1]
            integral.append(dist)
            throttle = dist * P_t + sum(integral) / len(integral) * I_t + derivative * D_t

            a[1] = throttle if throttle > 0. and speed < max_speed else 0.
            a[2] = abs(throttle) if throttle < 0. else 0.

            if self.log:
                print(
                    f'reward: {reward_total} speed: {speed:.2f}, steering: {a[0]:.2f}, throttle: {a[1]:.2f}, brake: {a[2]:.2f}')

            if self.plot:
                # visualise
                self.axes.clear()
                try:
                    self.axes.set_array(sdf.T)
                except AttributeError:
                    self.axes.imshow(sdf.T)
                line = torch.stack((origins, end), dim=-1)
                for l in range(line.shape[0]):
                    color = 'blue'
                    if l == i:
                        color = 'red'
                    self.axes.plot(line[l, 0], line[l, 1], color=color)

                plt.pause(0.01)

            if self.render:
                self.env.render()

            if done and self.demo:
                self.env.reset()
                a = np.array([0.0, 0.0, 0.0])


        # complete, return result
        self.env.close()
        return reward_total


class MpEvaluator(Evaluator):
    def __init__(self, num_workers):
        super().__init__()
        self.num_workers = num_workers

    def __call__(self, params, demo=False):
        worker_args = [param.numpy() for param in torch.unbind(params, dim=0)]

        with mp.Pool(processes=self.num_workers) as pool:
            results = list(tqdm(pool.imap(self.episode, worker_args), total=len(worker_args)))

        results = torch.tensor(results)
        return results


if __name__ == '__main__':
    eval = Evaluator(render=True, plot=True, log=True, demo=True)

    """
    t -> throttle, s -> steering
    params = [P_t, I_t, D_t, P_s, I_s, D_s, max_speed]
    max speed will be multiplied by 100. ie: 0.54 = max speed of 54
    """

    params = [ 0.3293,  0.2556,  0.3512, -0.4759,  0.8130, -0.0873,  0.5454]
    params = [-0.3982,  0.6430,  0.1186,  0.0183,  0.6377, -0.1834,  0.6460]
    params = [0.2398, 0.4720, 0.2737, 0.2110, 0.3450, 0.0845, 0.5262]
    params = [0.4867, 0.8937, 0.0142, 0.2889, 0.5399, 0.3729, 0.5422]
    params = [0.8282, 1.3217, -0.0188, -0.0594, 0.5394, 0.6987, 0.6252]
    params = [ 0.5290,  1.4023, -0.0398,  0.1445,  0.4489,  0.6587,  0.5192]
    params = [0.5172, 1.2380, 0.0180, 0.0952, 0.4438, 0.6632, 0.5539]
    # params = [0.5000, 1.2762, -0.0147, 0.1130, 0.4614, 0.6865, 0.5519]
    sdf, origins, end = eval.episode(params)
