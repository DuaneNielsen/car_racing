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
axes = fig.subplots(1, 4)
raw_plot, segment_plot, gradient_plot, sdf_plot = axes
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

    # Scharr the edges
    gradient_x = cv2.Sobel(segment, 3, dx=1, dy=0)
    gradient_y = cv2.Sobel(segment, 3, dx=0, dy=1)
    gradient = cv2.addWeighted(gradient_x, 0.5, gradient_y, 0.5, 0.0)
    gradient_plot.imshow(gradient)

    # Distance Field
    sdf = distance_transform_edt(gradient == 0)
    sdf_plot.imshow(sdf)

    env.render()
    fig.canvas.draw()
    episode_sdf += [sdf]
    episode_state += [state]
    x, y, theta = info['pos']
    episode_gt += [np.array([x, y, theta])]
    if len(episode_sdf) > 300:
        break


np.save('episode_sdf', np.stack(episode_sdf))
np.save('episode_state', np.stack(episode_state))
np.save('episode_gt', np.stack(episode_gt))
