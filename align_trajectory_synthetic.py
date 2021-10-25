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
    scatter_pos.scatter(x, y, color='blue')
    scatter_angle.scatter(timestep, theta, color='blue')


def draw_estimate(timestep, t1, rms):
    scatter_pos.scatter(t1.x, t1.y, color='red')
    scatter_angle.scatter(timestep, t1.theta, color='red')
    error_plt.scatter(timestep, rms, color='blue')


def plot_geo_diag(ax, kp0, kp1, t0, t1, space='t0'):
    ax.clear()
    ax.invert_yaxis()
    if space == 't0':
        kp0 = geo.transform_points(t0.inv_M, kp0)
        kp1 = geo.transform_points(t0.inv_M, kp1)
        t1_verts = geo.transform_points(t0.inv_M, t1.vertices_w)
        ax.add_patch(Polygon(t0.vertices.T, color='blue', fill=False))
        ax.add_patch(Polygon(t1_verts.T, color='red', fill=False))

    ax.scatter(kp0[0], kp0[1], label='t0', color='blue')
    ax.scatter(kp1[0], kp1[1], label='t1', color='red')
    centroid_t0 = kp0.mean(axis=1)
    centroid_t1 = kp1.mean(axis=1)
    ax.plot(*np.stack([centroid_t0, centroid_t1], axis=1))
    ax.legend()


def draw_stiched(ax, M, t0, t1):
    ax.clear()
    offset = np.zeros((2, 3))
    offset[0, 2] = 4.0
    offset[1, 2] = 4.0
    t0_warped = cv2.warpAffine(t0.image, np.eye(3)[0:2] + offset, (120, 70))
    t1_warped = cv2.warpAffine(t1.image, M[0:2] + offset, (120, 70))
    blend = cv2.addWeighted(t0_warped, 0.5, t1_warped, 0.5, 0.0)
    ax.imshow(blend)


def draw_readout(ax, text):
    ax.clear()
    ax.text(0.5, 0.67, text)


class RecordingEnv:
    def __init__(self, start=0, frameskip=0):
        # load captured data
        self.episode = np.load('episode_sdf.npy')
        self.episode_state = np.load('episode_state.npy')
        self.episode_gt = np.load('episode_gt.npy')
        self.episode_gt[:, 2] = -self.episode_gt[:, 2]
        self.h, self.w = self.episode.shape[1], self.episode.shape[2]
        self.start = start
        self.t = start
        if frameskip > 0:
            idx = np.arange(self.episode.shape[0]//3) * 3
            self.episode = self.episode[idx]
            self.episode_gt = self.episode_gt[idx]
            self.episode_state = self.episode_state[idx]

    def reset(self):
        self.t = self.start
        info = {'gt': self.episode_gt[self.t], 'state': self.episode_state[self.t]}
        return self.episode[self.t], info

    def step(self):
        self.t += 1
        done = self.t == len(self.episode) - 1
        info = {'gt': self.episode_gt[self.t], 'state': self.episode_state[self.t]}
        return self.episode[self.t], done, info


if __name__ == '__main__':

    # plotting
    fig = plt.figure(figsize=(18, 10))
    axes = fig.subplots(3, 3)
    state0_plt, state1_plt, text_plt = axes[0]
    unaligned, aligned, stitched = axes[1]
    unaligned.set_aspect('equal')
    aligned.set_aspect('equal')
    scatter_pos, scatter_angle, error_plt = axes[2]
    fig.show()

    env = RecordingEnv(start=76)
    scan_image, extras = env.reset()

    x, y, theta = extras['gt'][0], extras['gt'][1], extras['gt'][2]

    t1 = geo.Scan(env.h, env.w, scan_image, x, y, theta)

    t1_state = None
    if 'state' in extras:
        t1_state = extras['state']

    # list of sample indices
    h, w = env.h, env.w
    grid = geo.grid_sample(h, w, grid_spacing=16, pad=6)

    info = {}
    done = False
    timestep = 0
    trajectory, sdf, state = [t1.M], [t1.image], [t1_state]

    while not done:
        t0_state = t1_state
        t0 = t1

        t1_scan, done, extras = env.step()
        if 'state' in extras:
            t1_state = extras['state']

        t1 = geo.Scan(env.h, env.w, image=t1_scan)
        t1.R = t0.R
        t1.t = t0.t
        gt_pos = extras['gt']

        # t0.kp, t1.kp = env.step(0.5, 1.5, np.radians(1.0))
        # env.render()

        for _ in range(5):

            # svf keypoints
            grid_t0, grid_t1 = geo.project_and_clip_sample_indices(grid, t0, t1)
            t0.kp = extract_kp(t0.image, grid_t0, iterations=3)
            t1.kp = extract_kp(t1.image, grid_t1, iterations=3)

            # orb keypoints
            # orb = cv2.ORB_create(nfeatures=100, patchSize=5, edgeThreshold=3, fastThreshold=1)
            # sift = cv2.SIFT_create(nfeatures=100, contrastThreshold=0.01)
            # t0_kp, t1_kp = extract_cv2_kp(sift, state0, state1, match_metric=cv2.CV_32F)

            plot_geo_diag(unaligned, t0.kp_w, t1.kp_w, t0, t1)

            # filter keypoints
            t0_kp_w, t1_kp_w, filter = icp.filter_kp(t0.kp_w, t0.vertices_w, t1.kp_w, t1.vertices_w)
            rms_before = icp.rms(t1_kp_w, t0_kp_w)

            # compute alignment and update the t1 frame
            M, inliers, rms = icp.ransac_icp(source=t1_kp_w, target=t0_kp_w, k=19, n=3, threshold=3.0, d=5.0)
            t1.M = np.matmul(M, t1.M)

            draw_stiched(stitched, M, t0, t1)

            draw_state(state0_plt, t0.image, kp=geo.transform_points(t0.inv_M, t0_kp_w), inliers=inliers)
            draw_state(state1_plt, t1.image, kp=geo.transform_points(t1.inv_M, t1_kp_w), inliers=inliers)
            plot_geo_diag(aligned, t0.kp_w[:, filter][:, inliers], t1.kp_w[:, filter][:, inliers], t0, t1)
            draw_readout(text_plt, f'{timestep}')
            #plt.pause(2.0)
            plt.pause(0.05)

        draw_gt(*gt_pos)
        draw_estimate(timestep, t1, rms)
        print(f'timestep: {timestep}: rms: {rms} rms_before: {rms_before} '
              f't: {M[0:2,2:].squeeze()} R: {geo.theta(M[0:2,0:2])}')
        trajectory.append(t1.M)
        sdf.append(t1.image)
        state.append(t1_state)

        timestep += 1

    i = 0
    np.save(f'data/ep{i}_pose', np.stack(trajectory))
    np.save(f'data/ep{i}_sdf', np.stack(sdf))
    np.save(f'data/ep{i}_state', np.stack(state))
    plt.show()