import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import icp
import geometry as geo
from geometry import project_and_clip
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

t0_frame = geo.Scan(*t0.shape)
t1_frame = geo.Scan(*t1.shape)
# t1_frame.x += 5.0
# t1_frame.y += 10.0


def draw_scan_in_world(scan, color):
    scan_v_wf = geo.transform_points(scan.M, scan.vertices)
    world.add_patch(Polygon(scan_v_wf.T, color=color, fill=False))
    world.autoscale_view()


def draw_grid_in_world(grid, scan, label=None):
    grid_w = geo.transform_points(scan.M, grid)
    world.scatter(grid_w[0], grid_w[1], label=label)


def draw_frame():
    h, w = 100, 120
    world.clear()
    world.set_aspect('equal')

    world.imshow(cv2.warpAffine(t0, t0_frame.M[0:2], (w, h)), cmap='cool')
    draw_scan_in_world(t0_frame, color=[0, 0, 1.])
    draw_scan_in_world(t1_frame, color=[0, 1, 0.])
    plt.pause(1.0)
    world.imshow(cv2.warpAffine(t1, t1_frame.M[0:2], (w, h)), cmap='cool')
    draw_scan_in_world(t0_frame, color=[0, 0, 1.])
    draw_scan_in_world(t1_frame, color=[0, 1, 0.])
    plt.pause(1.0)


def draw():

    draw_frame()
    draw_grid_in_world(grid_t0, t0_frame, label='frame0')
    draw_grid_in_world(grid_t1, t1_frame, label='frame1')
    plt.legend()
    plt.pause(1.0)

    draw_frame()
    world.scatter(t0_kp_w[0], t0_kp_w[1], label='frame0')
    world.scatter(t1_kp_w[0], t1_kp_w[1], label='frame1')
    plt.legend()
    plt.pause(1.0)


for _ in range(20):

    # compute projections between the frames
    M_t0_t1 = np.matmul(t1_frame.inv_M, t0_frame.M)
    M_t1_t0 = np.matmul(t0_frame.inv_M, t1_frame.M)

    # project the t0 sample grid into t1s frame
    grid_t1 = np.round(geo.transform_points(M_t0_t1, grid)).astype(int)

    # filter sample points that are outside the t1 frame
    grid_t1_s_inside = geo.inside(grid_t1, t1_frame.vertices)
    grid_t1 = grid_t1[:, grid_t1_s_inside]
    grid_t0 = grid[:, grid_t1_s_inside]

    # extract key-points by following signed vector gradients
    t0_kp = extract_kp(t0, grid_t0)
    t1_kp = extract_kp(t1, grid_t1)

    t0_kp_w, t1_kp_w = project_and_clip(t0_kp, t1_kp, t0_frame, t1_frame)

    print(icp.rms(t1_kp_w, t0_kp_w))
    draw()

    # compute alignment and update the t1 frame
    R, t = icp.icp(t1_kp_w, t0_kp_w)
    t1_frame.t += t
    t1_frame.R = np.matmul(R, t1_frame.R)

plt.show()
