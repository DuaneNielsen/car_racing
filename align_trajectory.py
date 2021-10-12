import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import icp
import geometry as geo
from keypoints import extract_kp

"""
Estimate the poses of an ordered list of scan images
"""


def draw_scan_in_world(scan, color):
    scan_v_wf = geo.transform_points(scan.M, scan.vertices)
    world.add_patch(Polygon(scan_v_wf.T, color=color, fill=False))
    world.autoscale_view()


def draw_grid_in_world(grid, scan, label=None):
    grid_w = geo.transform_points(scan.M, grid)
    world.scatter(grid_w[0], grid_w[1], label=label)


def draw_scan(delay):
    h, w = 200, 200
    world.clear()
    world.set_aspect('equal')

    world.imshow(cv2.warpAffine(t0.image, t0.M[0:2], (w, h)), cmap='cool')
    draw_scan_in_world(t0, color=[0, 0, 1.])
    draw_scan_in_world(t1, color=[0, 1, 0.])
    plt.pause(delay)
    world.imshow(cv2.warpAffine(t1.image, t1.M[0:2], (w, h)), cmap='cool')
    draw_scan_in_world(t0, color=[0, 0, 1.])
    draw_scan_in_world(t1, color=[0, 1, 0.])
    plt.pause(delay)


def draw_oriented_scan(scan, color):
    scan_v_wf = geo.transform_points(scan.M, scan.vertices)
    world_traj.add_patch(Polygon(scan_v_wf.T, color=color, fill=False))
    world_traj.autoscale_view()
    world_traj.set_aspect('equal')


def draw(delay=0.05):

    # draw_scan(delay)
    # draw_grid_in_world(grid_t0, t0, label='scan t0')
    # draw_grid_in_world(grid_t1, t1, label='scan t1')
    # world.legend()
    #
    # plt.pause(delay)

    draw_scan(delay)
    state.imshow(state0)
    world.scatter(t0_kp_w[0], t0_kp_w[1], label='scan t0')
    world.scatter(t1_kp_w[0], t1_kp_w[1], label='scan t1')
    world.legend()
    plt.pause(delay)


if __name__ == '__main__':

    fig = plt.figure(figsize=(18, 10))
    axes = fig.subplots(1, 3)
    world, world_traj, state = axes
    world_traj.invert_yaxis()
    fig.show()

    episode = np.load('episode_sdf.npy')
    episode_state = np.load('episode_state.npy')

    trajectory = [geo.Scan(*step.shape, step) for step in episode]

    # list of sample indices
    grid = geo.grid_sample(*episode.shape[1:3], grid_spacing=16, pad=6)

    start = 70

    for timestep in range(start, len(trajectory)-1):

        state0 = episode_state[timestep]
        t0 = trajectory[timestep]
        t1 = trajectory[timestep + 1]
        t1.R = t0.R
        t1.t = t0.t

        draw_oriented_scan(t0, color=[0, 0, 1.])
        print(f'timestep: {timestep}')

        for _ in range(1):

            # project t0 sample index to t1 and filter points in the reference grid
            grid_t0, grid_t1 = geo.project_and_clip_sample_indices(grid, t0, t1)

            # extract key-points by following signed vector gradients
            t0_kp = extract_kp(t0.image, grid_t0)
            t1_kp = extract_kp(t1.image, grid_t1)

            # project key-points to world space and clip key-points outside the scan overlap
            t0_kp_w, t1_kp_w = geo.project_and_clip_kp(t0_kp, t1_kp, t0, t1)
            rms_before = icp.rms(t1_kp_w, t0_kp_w)

            # filter non unique key-points
            unique = geo.naive_unique(t0_kp_w)
            t0_kp_w, t1_kp_w = t0_kp_w[:, unique], t1_kp_w[:, unique]
            unique = geo.naive_unique(t1_kp_w)
            t0_kp_w, t1_kp_w = t0_kp_w[:, unique], t1_kp_w[:, unique]

            # compute alignment and update the t1 frame
            R, t, t1_kp_w, t0_kp_w, error = icp.ransac_icp(t1_kp_w, t0_kp_w, k=12, n=7)
            print(f'rms_before {rms_before},  rms_after: {error}')

            t1.t += t
            t1.R = np.matmul(R, t1.R)

        #draw(0.01)

    traj_pose = np.stack([s.M for s in trajectory])
    i = 0
    np.save(f'data/ep{i}_pose', traj_pose[start:])
    np.save(f'data/ep{i}_sdf', episode[start:])
    np.save(f'data/ep{i}_state', episode_state[start:])
    plt.show()
