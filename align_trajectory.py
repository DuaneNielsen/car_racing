import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import icp
import geometry as geo
from keypoints import extract_kp


def draw(delay=0.05):
    state.imshow(state0)

    scatter_pos.scatter(gt_pos[0], gt_pos[1], color='red')
    scatter_pos.scatter(t0.x, t0.y, color='blue')
    scatter_angle.scatter(timestep, gt_pos[2], color='red')
    scatter_angle.scatter(timestep, t0.theta, color='blue')
    error_plt.scatter(timestep, info['rms'], color='blue')
    plt.pause(delay)


def plot_kp(ax, kp, label=None):
    ax.scatter(kp[0], kp[1])
    ax.scatter(*np.split(kp.mean(axis=1), 2), label=label)


def plot_unaligned(t0_image, t0_kp, t1_kp):
    unaligned.clear()
    unaligned.imshow(t0_image)
    plot_kp(unaligned, t0_kp, label='t0_kp_centroid')
    plot_kp(unaligned, t1_kp, label='t1_kp_centroid')
    unaligned.legend()


def plot_aligned(t1_image, source_kp, target_kp, t, R):
    aligned.clear()
    aligned.imshow(t1_image)
    plot_kp(aligned, np.matmul(R, source_kp + t))
    plot_kp(aligned, target_kp)


if __name__ == '__main__':

    fig = plt.figure(figsize=(18, 10))
    axes = fig.subplots(2, 3)
    unaligned, aligned, state = axes[0]
    scatter_pos, scatter_angle, error_plt = axes[1]
    fig.show()

    episode = np.load('episode_sdf.npy')
    episode_state = np.load('episode_state.npy')
    episode_gt = np.load('episode_gt.npy')

    trajectory = [geo.Scan(*step.shape, step) for step in episode]

    # list of sample indices
    grid = geo.grid_sample(*episode.shape[1:3], grid_spacing=4, pad=6)

    filter_N_0 = grid.shape[1]

    start = 160

    # initialize t0 to be same as ground truth
    trajectory[start].x = episode_gt[start, 0]
    trajectory[start].y = episode_gt[start, 1]
    trajectory[start].theta = episode_gt[start, 2]

    for timestep in range(start, len(trajectory)-1):

        state0 = episode_state[timestep]
        t0 = trajectory[timestep]
        t1 = trajectory[timestep + 1]
        t1.R = t0.R.copy()
        t1.t = t0.t.copy()
        gt_pos = episode_gt[timestep]
        gt = episode_gt[timestep + 1] - episode_gt[timestep]

        print(f'timestep: {timestep}')

        for _ in range(1):

            # project t0 sample index to t1 and filter points in the reference grid
            grid_t0, grid_t1 = geo.project_and_clip_sample_indices(grid, t0, t1)

            filter_N_project_and_clip = grid_t0.shape[1]

            # extract key-points by following signed vector gradients
            t0_kp = extract_kp(t0.image, grid_t0, iterations=1)
            t1_kp = extract_kp(t1.image, grid_t1, iterations=1)

            t1, R, t, info = icp.update_frame(t0, t1, t0_kp, t1_kp, k=20, n=20)

            plot_unaligned(t0.image, info['t0_kp'], info['t1_kp'])
            plot_aligned(t1.image, info['t0_kp'], info['t1_kp'], t, R)

        draw(5.00)

    traj_pose = np.stack([s.M for s in trajectory])
    i = 0
    np.save(f'data/ep{i}_pose', traj_pose[start:])
    np.save(f'data/ep{i}_sdf', episode[start:])
    np.save(f'data/ep{i}_state', episode_state[start:])
    np.save(f'data/ep{i}_gt', episode_gt[start:])
    plt.show()