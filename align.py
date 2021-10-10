import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import icp
import geometry as geo
from keypoints import extract_kp

fig = plt.figure(figsize=(18, 10))
axes = fig.subplots(1)
world = axes
fig.show()

episode = np.load('episode.npy')

# list of sample indices
grid = geo.grid_sample(*episode.shape[1:3], grid_spacing=16, pad=6)

timestep = 35

t0 = episode[timestep]
t1 = episode[timestep + 1]

t0_scan = geo.Scan(*t0.shape)
t1_scan = geo.Scan(*t1.shape)


def draw_scan_in_world(scan, color):
    scan_v_wf = geo.transform_points(scan.M, scan.vertices)
    world.add_patch(Polygon(scan_v_wf.T, color=color, fill=False))
    world.autoscale_view()


def draw_grid_in_world(grid, scan, label=None):
    grid_w = geo.transform_points(scan.M, grid)
    world.scatter(grid_w[0], grid_w[1], label=label)


def draw_scan():
    h, w = 100, 120
    world.clear()
    world.set_aspect('equal')

    world.imshow(cv2.warpAffine(t0, t0_scan.M[0:2], (w, h)), cmap='cool')
    draw_scan_in_world(t0_scan, color=[0, 0, 1.])
    draw_scan_in_world(t1_scan, color=[0, 1, 0.])
    plt.pause(1.0)
    world.imshow(cv2.warpAffine(t1, t1_scan.M[0:2], (w, h)), cmap='cool')
    draw_scan_in_world(t0_scan, color=[0, 0, 1.])
    draw_scan_in_world(t1_scan, color=[0, 1, 0.])
    plt.pause(1.0)


def draw():

    draw_scan()
    draw_grid_in_world(grid_t0, t0_scan, label='scan t0')
    draw_grid_in_world(grid_t1, t1_scan, label='scan t1')
    plt.legend()
    plt.pause(1.0)

    draw_scan()
    world.scatter(t0_kp_w[0], t0_kp_w[1], label='scan t0')
    world.scatter(t1_kp_w[0], t1_kp_w[1], label='scan t1')
    plt.legend()
    plt.pause(1.0)


for _ in range(20):

    # project t0 sample index to t1 and filter points in the reference grid
    grid_t0, grid_t1 = geo.project_and_clip_sample_indices(grid, t0_scan, t1_scan)

    # extract key-points by following signed vector gradients
    t0_kp = extract_kp(t0, grid_t0)
    t1_kp = extract_kp(t1, grid_t1)

    # project key-points to world space and clip key-points outside the scan overlap
    t0_kp_w, t1_kp_w = geo.project_and_clip_kp(t0_kp, t1_kp, t0_scan, t1_scan)

    print(icp.rms(t1_kp_w, t0_kp_w))
    draw()

    # compute alignment and update the t1 frame
    R, t = icp.icp(t1_kp_w, t0_kp_w)
    t1_scan.t += t
    t1_scan.R = np.matmul(R, t1_scan.R)

plt.show()
