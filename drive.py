import gym
from matplotlib import pyplot as plt
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from pyglet.window import key
import env

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


fig = plt.figure(figsize=(18, 10))
axes = fig.subplots(1, 5)
raw_plot, segment_plot, gradient_plot, sdf_plot, road_plot = axes
fig.show()

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

while not done:
    state, reward, done, info = env.step(a)
    print(info)
    # action = env.action_space.sample()
    # action[1], action[2] = 0.5, 0.0  # gas, brake

    # crop the input image
    crop = state[:60]
    [ax.clear() for ax in axes]
    raw_plot.imshow(crop)

    # segment the image
    road = np.logical_and(crop[:, :, 1] > 100, crop[:, :, 1] < 110)
    bushes = crop[:, :, 1] == 229
    segment = (road | bushes).astype(float)
    segment_plot.imshow(segment)

    def gradient(segment):
        # Scharr the edges
        gradient_x = cv2.Sobel(segment, 3, dx=1, dy=0)
        gradient_y = cv2.Sobel(segment, 3, dx=0, dy=1)
        return cv2.addWeighted(gradient_x, 0.5, gradient_y, 0.5, 0.0)

    gradient_segment = gradient(segment)
    gradient_plot.imshow(gradient_segment)

    # Distance Field
    sdf = distance_transform_edt(gradient_segment == 0)
    sdf_plot.imshow(sdf)

    sdf_road = distance_transform_edt(gradient(road.astype(float)) == 0)
    road_plot.imshow(sdf_road)

    env.render()
    fig.canvas.draw()
    episode_sdf += [sdf]
    episode_sdf_road += [sdf_road]
    episode_state += [state]
    x, y, theta = info['pos']
    episode_gt += [np.array([x, y, theta])]
    if len(episode_sdf) > 1600:
        break

i = 3
np.save(f'ep{i}_sdf', np.stack(episode_sdf))
np.save(f'ep{i}_state', np.stack(episode_state))
np.save(f'ep{i}_gt', np.stack(episode_gt))
np.save(f'ep{i}_sdf_road', np.stack(episode_sdf_road))