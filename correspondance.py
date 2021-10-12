from matplotlib import pyplot as plt
import numpy as np
import geometry as geo
from keypoints import extract_kp

"""
Extract corresponding key-points from a Signed Distance Field 
"""


fig = plt.figure(figsize=(18, 10))
axes = fig.subplots(1, 2)
t0_plot, t1_plot = axes
fig.show()

episode = np.load('episode.npy')

# list of sample indices
grid = geo.grid_sample(*episode.shape[1:3], grid_spacing=16, pad=2)

for t in range(30, episode.shape[0]-1):

    t0 = episode[t]
    t1 = episode[t + 1]

    [ax.clear() for ax in axes]
    t0_plot.imshow(t0)
    t1_plot.imshow(t1)

    t0_kp = extract_kp(t0, grid)
    t1_kp = extract_kp(t1, grid)

    # plot the corresponding points
    for i in range(t0_kp.shape[1]):
        t0_plot.scatter(t0_kp[0, i], t0_kp[1, i])
        t1_plot.scatter(t1_kp[0, i], t1_kp[1, i])

    fig.canvas.draw()
    plt.show()

