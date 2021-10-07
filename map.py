import gym
from matplotlib import pyplot as plt
import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt

fig = plt.figure(figsize=(18, 10))
axes = fig.subplots(1, 4)
raw_plot, segment_plot, gradient_plot, sdf_plot = axes
fig.show()

env = gym.make('CarRacing-v0')
state, done = env.reset(), False
action = env.action_space.sample()
env.render()

episode = []

while not done:
    state, reward, done, info = env.step(action)
    action = env.action_space.sample()
    action[1], action[2] = 0.5, 0.0  # gas, brake

    # crop the input image
    crop = state[:60]
    [ax.clear() for ax in axes]
    raw_plot.imshow(crop)

    # segment the image
    segment = np.logical_and(crop[:, :, 1] > 100, crop[:, :, 1] < 110).astype(float)
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
    episode += [sdf]
    if len(episode) > 300:
        break


np.save('episode', episode)