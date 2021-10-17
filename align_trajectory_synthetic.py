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
    h, w = 300, 300
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


def draw_relative_frames(f0, kp0_w, f1, kp1_w, R, t):
    relative.clear()
    kp0 = geo.transform_points(f0.inv_M, kp0_w)
    kp1 = geo.transform_points(f1.inv_M, kp1_w)
    relative.scatter(kp0[0], kp0[1])
    relative.scatter(kp1[0], kp1[1])
    centroid = kp0.mean(axis=1)
    relative.plot([centroid[0], centroid[0] + t[0]], [centroid[1], centroid[1] + t[1]])


def draw(delay=0.05):

    # draw_relative_frames(t0, t0_kp_w, t1, t1_kp_w, R, t)

    # draw_scan(delay)
    # draw_grid_in_world(grid_t0, t0, label='scan t0')
    # draw_grid_in_world(grid_t1, t1, label='scan t1')
    # world.legend()

    # plt.pause(delay)

    #draw_scan(delay)
    state.clear()
    state.imshow(state0)
    state.scatter(t1_kp[0], t1_kp[1])
    #world.scatter(t0_kp_w[0], t0_kp_w[1], label='scan t0')
    #world.scatter(t1_kp_w[0], t1_kp_w[1], label='scan t1')

    # draw gt vs track
    scatter_pos.scatter(gt_pos[0], gt_pos[1], color='red')
    scatter_pos.scatter(t0.x, t0.y, color='blue')
    scatter_angle.scatter(timestep, -gt_pos[2], color='red')
    scatter_angle.scatter(timestep, t0.theta, color='blue')
    error_plt.scatter(timestep, error, color='blue')
    #world.legend()
    plt.pause(delay)



def gt_images(gt, h, w):
    gridf0 = geo.grid_sample(h, w, 12, pad=2)
    R = geo.R(gt[2])
    gridf1 = np.matmul(R, gridf0) + gt[0:2].reshape(2, 1)
    return gridf0, gridf1


def plot_geo_diag(ax, kp0, kp1):
    ax.clear()
    ax.scatter(kp0[0], kp0[1])
    ax.scatter(kp1[0], kp1[1])
    centroid_t0 = kp0.mean(axis=1)
    centroid_t1 = kp1.mean(axis=1)
    ax.plot([0, centroid_t0[0]],  [0, centroid_t0[1]])
    ax.plot([centroid_t0[0], t[0]], [centroid_t0[1], t[1]])
    ax.plot([0, t[0]], [0, t[1]])


def plot_geo():
    kp0_w = geo.transform_points(t0.inv_M, t0_kp_t0)
    kp1_w = geo.transform_points(t1.inv_M, t1_kp_t1)
    plot_geo_diag(f0_geo_plt, t0_kp_t0, t1_kp_t1)
    plot_geo_diag(world_geo, kp0_w, kp1_w)


if __name__ == '__main__':

    fig = plt.figure(figsize=(18, 10))
    axes = fig.subplots(1, 6)
    world_geo, f0_geo_plt, state, scatter_pos, scatter_angle, error_plt = axes
    #world_traj.invert_yaxis()
    fig.show()

    episode = np.load('episode_sdf.npy')
    episode_state = np.load('episode_state.npy')
    episode_gt = np.load('episode_gt.npy')

    trajectory = [geo.Scan(*step.shape, step) for step in episode]

    # list of sample indice
    h, w = episode.shape[1:3]
    grid = geo.grid_sample(h, w, grid_spacing=12, pad=6)

    filter_N_0 = grid.shape[1]

    start = 140

    # initialize t0 to be same as ground truth
    trajectory[start].x = episode_gt[start, 0]
    trajectory[start].y = episode_gt[start, 1]
    trajectory[start].theta = episode_gt[start, 2]

    for timestep in range(start, len(trajectory)-1):

        state0 = episode_state[timestep]
        t0 = trajectory[timestep]
        t1 = trajectory[timestep + 1]
        t1.R = t0.R
        t1.t = t0.t
        gt_pos = episode_gt[timestep]
        gt = episode_gt[timestep + 1] - episode_gt[timestep]

        #draw_oriented_scan(t0, color=[0, 0, 1.])
        print(f'timestep: {timestep}')

        for _ in range(1):

            # project t0 sample index to t1 and filter points in the reference grid
            grid_t0, grid_t1 = geo.project_and_clip_sample_indices(grid, t0, t1)

            filter_N_project_and_clip = grid_t0.shape[1]

            # extract key-points by following signed vector gradients
            #t0_kp_orig = extract_kp(t0.image, grid_t0, iterations=1)
            #t1_kp_orig = extract_kp(t1.image, grid_t1, iterations=1)

            #t0_kp, t1_kp = gt_fake_scans[timestep].image, gt_fake_scans[timestep+1].image

            t0_kp, t1_kp = gt_images(gt, h, w)

            # project key-points to world space and clip key-points outside the scan overlap
            t0_kp_w, t1_kp_w = geo.project_and_clip_kp(t0_kp, t1_kp, t0, t1)
            rms_before = icp.rms(t1_kp_w, t0_kp_w)
            filter_n_intersect = t0_kp_w.shape[1]

            # filter non unique key-points
            unique = geo.naive_unique(t0_kp_w)
            t0_kp_w, t1_kp_w = t0_kp_w[:, unique], t1_kp_w[:, unique]
            unique = geo.naive_unique(t1_kp_w)
            t0_kp_w, t1_kp_w = t0_kp_w[:, unique], t1_kp_w[:, unique]
            filter_n_unique = t0_kp_w.shape[1]

            t0_kp_t0, t1_kp_t1 = geo.transform_points(t0.inv_M, t0_kp_w), geo.transform_points(t1.inv_M, t1_kp_w)

            kp_filters = f'start:{filter_N_0} proj:{filter_N_project_and_clip} intersect: {filter_n_intersect} ' \
                         f'unique: {filter_n_unique}'
            print(kp_filters)

            # compute alignment and update the t1 frame
            R, t, t0_kp_t0, t1_kp_t1, error = icp.ransac_icp(t0_kp_t0, t1_kp_t1, k=2, n=3)

            plot_geo()
            fig.canvas.draw()

            #t1.t = t1.t + np.matmul(R, t)
            t1.t = t1.t + t
            t1.R = np.matmul(R, t1.R)

            theta = np.arccos(R[0, 0])

            print(f'rms_before {rms_before},  rms_after: {error} t:{t.squeeze()} theta:{theta}')

        theta = np.array([np.arccos(t1.R[0, 0]) - np.arccos(t0.R[0, 0])])
        d_pose = np.concatenate([(t1.t - t0.t).squeeze(), theta])

        draw(0.01)

    traj_pose = np.stack([s.M for s in trajectory])
    i = 0
    np.save(f'data/ep{i}_pose', traj_pose[start:])
    np.save(f'data/ep{i}_sdf', episode[start:])
    np.save(f'data/ep{i}_state', episode_state[start:])
    np.save(f'data/ep{i}_gt', episode_gt[start:])
    plt.show()