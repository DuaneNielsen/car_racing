import gym
from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import distance_transform_edt
from pyglet.window import key
import argparse

from vision_utils import road_segment, gradient, road_distance_field

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

class Plot:
    def __init__(self, plot):
        self.plot = plot
        self.fig = plt.figure(figsize=(18, 10))
        self.axes = self.fig.subplots(1, 5)
        self.raw_plot, self.segment_plot, self.gradient_plot, self.sdf_plot, self.road_plot = self.axes
        if plot:
            self.fig.show()

    def imshow(self, ax, img):
        if self.plot:
            ax.imshow(img)

    def clear(self):
        if self.plot:
            for ax in self.axes:
                ax.clear()

parser = argparse.ArgumentParser()
parser.add_argument('ep', type=int)
parser.add_argument('--visualize', '-v', action='store_true', default=False)
args = parser.parse_args()

plot = Plot(args.visualize)

env = gym.make('CarRacing-v1')
state, done = env.reset(), False
action = env.action_space.sample()
env.render()
env.viewer.window.on_key_press = key_press
env.viewer.window.on_key_release = key_release

episode_sdf = []
episode_state = []
episode_gt = []
episode_sdf_road = []
episode_segment = []

while not done:
    state, reward, done, info = env.step(a)
    print(info)
    # action = env.action_space.sample()
    # action[1], action[2] = 0.5, 0.0  # gas, brake

    # crop the input image
    crop = state[:60]

    plot.imshow(plot.raw_plot, crop)

    # segment the image
    road = np.logical_and(crop[:, :, 1] > 100, crop[:, :, 1] < 110)
    bushes = crop[:, :, 1] == 229
    segment = (road | bushes).astype(float)
    plot.imshow(plot.segment_plot, segment)

    gradient_segment = gradient(segment)
    plot.imshow(plot.gradient_plot, gradient_segment)

    # Distance Field
    sdf = distance_transform_edt(gradient_segment == 0)
    plot.imshow(plot.sdf_plot, sdf)

    sdf_road = distance_transform_edt(gradient(road.astype(float)) == 0)
    sign_road = road * 2 - 1
    sdf_road = sdf_road * sign_road
    plot.imshow(plot.road_plot, road_distance_field(observation=state[:60], segment_func=road_segment)[0])

    env.render()
    plt.pause(0.01)
    episode_sdf += [sdf]
    episode_sdf_road += [sdf_road]
    episode_state += [state]
    episode_segment += [road]
    x, y, theta = info['pos']
    episode_gt += [np.array([x, y, theta])]
    if len(episode_sdf) > 1600:
        break


np.save(f'data/ep{args.ep}_sdf', np.stack(episode_sdf))
np.save(f'data/ep{args.ep}_state', np.stack(episode_state))
np.save(f'data/ep{args.ep}_gt', np.stack(episode_gt))
np.save(f'data/ep{args.ep}_sdf_road', np.stack(episode_sdf_road))
np.save(f'data/ep{args.ep}_segment', np.stack(episode_segment))
