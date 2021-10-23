import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import icp
import geometry as geo
from keypoints import extract_kp, extract_cv2_kp
import synthworld

"""
Estimate the poses of an ordered list of scan images
"""


def draw_state(ax, state=None, kp=None, inliers=None):
    ax.clear()
    if state is not None:
        ax.imshow(state)
    if inliers is not None:
        ax.scatter(kp[0, inliers], kp[1, inliers], color='blue')
        ax.scatter(kp[0, ~inliers], kp[1, ~inliers], color='red')
    else:
        ax.scatter(kp[0], kp[1], color='blue')


def draw_gt(x, y, theta):
    # draw gt vs track
    scatter_pos.scatter(x, y, color='red')
    scatter_angle.scatter(timestep, theta, color='red')


def draw_estimate():
    scatter_pos.scatter(t1.x, t1.y, color='blue')
    scatter_angle.scatter(timestep, t0.theta, color='blue')
    error_plt.scatter(timestep, info['rms'], color='blue')

def gt_images(gt, h, w):
    gridf0 = geo.grid_sample(h, w, 12, pad=2)
    R = geo.R(gt[2])
    gridf1 = np.matmul(R, gridf0) + gt[0:2].reshape(2, 1)
    return gridf0, gridf1


def plot_geo_diag(ax, kp0, kp1, t=None):
    ax.clear()
    ax.scatter(kp0[0], kp0[1], label='t0')
    ax.scatter(kp1[0], kp1[1], label='t1')
    centroid_t0 = kp0.mean(axis=1)
    centroid_t1 = kp1.mean(axis=1)
    #ax.scatter(centroid_t0[0], centroid_t0[1], label='t0 centroid')
    #ax.scatter(centroid_t1[0], centroid_t1[1], label='t1 centroid')
    #ax.quiver(*centroid_t0, *centroid_t1)
    if t is not None:
        ax.plot(*np.stack([centroid_t0, centroid_t0 + t.squeeze()], axis=1))
    else:
        ax.plot(*np.stack([centroid_t0, centroid_t1], axis=1))
    ax.legend()


def plot_geo(t0_kp_t0, t1_kp_t1):
    # kp0_w = geo.transform_points(t0.inv_M, t0_kp_t0)
    # kp1_w = geo.transform_points(t1.inv_M, t1_kp_t1)
    plot_geo_diag(aligned, t0_kp_t0, t1_kp_t1)
    #plot_geo_diag(world_geo, kp0_w, kp1_w)


if __name__ == '__main__':

    fig = plt.figure(figsize=(18, 10))
    axes = fig.subplots(3, 3)
    state0_plt, state1_plt, _ = axes[0]
    unaligned, aligned, _ = axes[1]
    scatter_pos, scatter_angle, error_plt = axes[2]
    #world_traj.invert_yaxis()
    fig.show()

    episode = np.load('episode_sdf.npy')
    episode_state = np.load('episode_state.npy')
    episode_gt = np.load('episode_gt.npy')

    idx = np.arange(episode.shape[0]//3) * 3
    start = 70 // 3

    episode = episode[idx]
    episode_gt = episode_gt[idx]
    episode_state = episode_state[idx]

    trajectory = [geo.Scan(*step.shape, step) for step in episode]

    # list of sample indice
    h, w = episode.shape[1:3]
    grid = geo.grid_sample(h, w, grid_spacing=16, pad=6)



    # initialize t0 to be same as ground truth
    trajectory[start].x = episode_gt[start, 0]
    trajectory[start].y = episode_gt[start, 1]
    trajectory[start].theta = episode_gt[start, 2]

    env = synthworld.World(x=200.0, y=200)
    env.reset()
    env.render()

    trajectory[start].x = env.f.x
    trajectory[start].y = env.f.y
    trajectory[start].theta = env.f.theta

    for timestep in range(start, len(trajectory)-1):

        state0, state1 = episode_state[timestep][:60], episode_state[timestep + 1][:60]
        t0 = trajectory[timestep]
        t1 = trajectory[timestep + 1]
        t1.R = t0.R
        t1.t = t0.t
        gt_pos = episode_gt[timestep]
        gt = episode_gt[timestep + 1] - episode_gt[timestep]

        t0_kp, t1_kp = env.step(0.5, 1.5, np.radians(1.0))
        env.render()
        plot_geo_diag(unaligned, t0_kp, t1_kp)
        draw_gt(env.f.x, env.f.y, env.f.theta)
        plt.pause(0.04)

        for _ in range(1):


            # svf keypoints
            # grid_t0, grid_t1 = geo.project_and_clip_sample_indices(grid, t0, t1)
            # t0_kp = extract_kp(t0.image, grid_t0, iterations=1)
            # t1_kp = extract_kp(t1.image, grid_t1, iterations=1)

            # orb keypoints
            # orb = cv2.ORB_create(nfeatures=100, patchSize=5, edgeThreshold=3, fastThreshold=1)
            # sift = cv2.SIFT_create(nfeatures=100, contrastThreshold=0.01)
            # t0_kp, t1_kp = extract_cv2_kp(sift, state0, state1, match_metric=cv2.CV_32F)

            # synthetic keypoints
            #t0_kp, t1_kp = gt_images(gt, h, w)

            t1, R, t, info = icp.update_frame(t0, t1, t0_kp, t1_kp, k=10, n=3, threshold=12.0, d=1.0)

            inliers, t0_kp, t1_kp = info['inliers'], info['t0_kp'], info['t1_kp']
            draw_estimate()

            draw_state(state0_plt, kp=t0_kp, inliers=inliers)
            draw_state(state1_plt, kp=t1_kp, inliers=inliers)
            #plot_geo_diag(unaligned, info['t0_kp'], info['t1_kp'])
            plot_geo_diag(aligned, np.matmul(R, info['t0_kp'] + t), info['t1_kp'], t)


        theta = np.array([np.arccos(t1.R[0, 0]) - np.arccos(t0.R[0, 0])])
        rms, rms_before, = info['rms'], info['rms_before']
        print(f'timestep: {timestep}: rms: {rms} rms_before: {rms_before} t: {t.squeeze()} R: {geo.theta(R)}')

    traj_pose = np.stack([s.M for s in trajectory])
    i = 0
    np.save(f'data/ep{i}_pose', traj_pose[start:])
    np.save(f'data/ep{i}_sdf', episode[start:])
    np.save(f'data/ep{i}_state', episode_state[start:])
    np.save(f'data/ep{i}_gt', episode_gt[start:])
    plt.show()