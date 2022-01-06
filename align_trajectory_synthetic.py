import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import icp
import geometry as geo
import keypoints as kps
from keypoints import extract_kp, extract_cv2_kp
import synthworld
import argparse

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
    def __init__(self, episode, start=0, frameskip=0):
        # load captured data
        self.episode = np.load(f'data/ep{episode}_sdf.npy')
        self.episode_road = np.load(f'data/ep{episode}_sdf_road.npy')
        self.episode_state = np.load(f'data/ep{episode}_state.npy')
        self.episode_gt = np.load(f'data/ep{episode}_gt.npy')
        self.episode_segment = np.load(f'data/ep{episode}_segment.npy')
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
        info = {'gt': self.episode_gt[self.t], 'state': self.episode_state[self.t], 'road': self.episode_road[self.t]}
        return self.episode[self.t], info

    def step(self):
        self.t += 1
        done = self.t == len(self.episode) - 1
        info = {'gt': self.episode_gt[self.t], 'state': self.episode_state[self.t], 'road': self.episode_road[self.t]}
        return self.episode[self.t], done, info


def make_grid(h, w, homo=False):
    axis = []
    axis += [*np.meshgrid(np.arange(h), np.arange(w))]
    if homo:
        axis += [np.ones_like(axis[0])]
    return np.stack(axis, axis=2)


if __name__ == '__main__':

    """
    you can't re- run this as it will cut off the first 70 frames eaach run.
    # todo fix this
    """

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("-ep", "--episode", type=int, default=0)
    parser.add_argument('-v', "--visualize", action='store_true', default=False)
    args = parser.parse_args()

    # plotting
    if args.visualize:
        fig = plt.figure(figsize=(18, 10))
        axes = fig.subplots(3, 3)
        state0_plt, state1_plt, text_plt = axes[0]
        unaligned, aligned, stitched = axes[1]
        unaligned.set_aspect('equal')
        aligned.set_aspect('equal')
        scatter_pos, scatter_angle, error_plt = axes[2]
        fig.show()

        fig_world = plt.figure()
        world_plot = fig_world.subplots(1, 1)

    env = RecordingEnv(args.episode, start=70)
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
    #trajectory, trajectory_d, sdf, state, errors = [t1.M], [np.eye(3)], [t1.image], [t1_state], [0.]
    trajectory, trajectory_d, sdf, state, errors = [np.eye(3)], [np.eye(3)], [t1.image], [t1_state], [0.]

    # world array
    world = np.full((2000, 2000), 4.0)
    world_n = np.zeros_like(world)
    world_origin = np.array([t1.x.item() + world.shape[0]//2, t1.y.item() + world.shape[1]//2])
    max_h, min_h, max_w, min_w = 0, world.shape[1], 0, world.shape[0]

    while not done:
        t0_state = t1_state
        t0 = t1

        t1_scan, done, extras = env.step()
        if 'state' in extras:
            t1_state = extras['state']

        t1 = geo.Scan(env.h, env.w, image=t1_scan)
        t1.road = extras['road']
        t1.R = t0.R
        t1.t = t0.t
        gt_pos = extras['gt']

        # t0.kp, t1.kp = env.step(0.5, 1.5, np.radians(1.0))
        # env.render()
        delta = np.eye(3)

        for _ in range(5):

            # svf keypoints
            grid_t0, grid_t1 = geo.project_and_clip_sample_indices(grid, t0, t1)
            t0.kp = extract_kp(t0.image, grid_t0, iterations=3)
            t1.kp = extract_kp(t1.image, grid_t1, iterations=3)

            # orb keypoints
            # orb = cv2.ORB_create(nfeatures=100, patchSize=5, edgeThreshold=3, fastThreshold=1)
            # sift = cv2.SIFT_create(nfeatures=100, contrastThreshold=0.01)
            # t0_kp, t1_kp = extract_cv2_kp(sift, state0, state1, match_metric=cv2.CV_32F)

            if args.visualize:
                unaligned.clear()
                t1.draw(unaligned, color='red', label='t1')
                t0.draw(unaligned, items=['box', 'kp'], color='blue', label='t0')


            # filter keypoints
            t0_kp_w, t1_kp_w, filter = icp.filter_kp(t0.kp_w, t0.vertices_w, t1.kp_w, t1.vertices_w)
            rms_before = icp.rms(t1_kp_w, t0_kp_w)

            # compute alignment and update the t1 frame
            M, t, R, inliers, rms = icp.ransac_icp(source=t1_kp_w, target=t0_kp_w, k=19, n=3, threshold=3.0, d=5.0)
            t1.M = np.matmul(M, t1.M)
            delta = np.matmul(M, delta)

            if args.visualize:
                draw_stiched(stitched, M, t0, t1)
                draw_state(state0_plt, t0.image, kp=geo.transform_points(t0.inv_M, t0_kp_w), inliers=inliers)
                draw_state(state1_plt, t1.image, kp=geo.transform_points(t1.inv_M, t1_kp_w), inliers=inliers)
                plot_geo_diag(aligned, t0.kp_w[:, filter][:, inliers], t1.kp_w[:, filter][:, inliers], t0, t1)
                draw_readout(text_plt, f'{timestep}')
                #plt.pause(0.5)

        # write to world array
        image = t1.road.clip(-4.0, 4.0)
        mask = (image > -4.0) & (image < 4.0)
        model_grid = make_grid(w, h, homo=True).reshape(h * w, 3).T
        model_grid = model_grid[:, mask.reshape(h * w)]
        model_grid_w = np.matmul(t1.M, model_grid)[0:2] + world_origin.reshape(2, 1)
        model_grid_w = model_grid_w.round().astype(np.int64)
        h_i, w_i = model_grid_w[0], model_grid_w[1]
        n = world_n[w_i, h_i]
        world[w_i, h_i] = ((world[w_i, h_i] * n + image[model_grid[1], model_grid[0]]) / (n + 1))
        world_n[w_i, h_i] += 1.

        if args.visualize:
            world_plot.clear()
            max_h, min_h = max(h_i.max(), max_h), min(h_i.min(), min_h)
            max_w, min_w = max(w_i.max(), max_w), min(w_i.min(), min_w)
            world_plot.imshow(world[min_w:max_w, min_h:max_h], cmap='cool')
        kp = t1.kp_w + world_origin.reshape(2, 1)

        #draw_gt(*gt_pos)
        #draw_estimate(timestep, t1, rms)
        print(f'timestep: {timestep}: rms: {rms} rms_before: {rms_before} '
              f't: {M[0:2,2:].squeeze()} R: {geo.theta(M[0:2,0:2])}')
        trajectory.append(t1.M)
        trajectory_d.append(delta)
        sdf.append(t1.image)
        state.append(t1_state)
        errors.append(rms)
        plt.pause(0.05)
#
        timestep += 1

    np.save(f'data/ep{args.episode}_sdf.npy', env.episode[env.start:])
    np.save(f'data/ep{args.episode}_sdf_road.npy', env.episode_road[env.start:])
    np.save(f'data/ep{args.episode}_state.npy', env.episode_state[env.start:])
    np.save(f'data/ep{args.episode}_gt.npy', env.episode_gt[env.start:])
    np.save(f'data/ep{args.episode}_segment.npy', env.episode_segment[env.start:])

    np.save(f'data/ep{args.episode}_map', world)
    np.save(f'data/ep{args.episode}_pose', np.stack(trajectory))
    np.save(f'data/ep{args.episode}_pose_d', np.stack(trajectory_d))
    np.save(f'data/ep{args.episode}_error', np.stack(errors))
    #plt.show()


class Episode:
    def __init__(self, i):
        self.sdfs = np.load(f'data/ep{i}_sdf_road.npy')[70:]
        self.states = np.load(f'data/ep{i}_state.npy')[70:]
        self.pose = np.load(f'data/ep{i}_pose.npy')[70:]
        self.pose_d = np.load(f'data/ep{i}_pose_d.npy')[70:]
        self.map = np.load(f'data/ep{i}_map.npy')
        self.rms = np.load(f'data/ep{i}_error.npy')
        self.N, self.h, self.w = self.sdfs.shape
        sample_i = geo.grid_sample(self.h, self.w, 12, pad=4)
        keypoints = []
        for i, sdf in enumerate(self.sdfs):
            keypoints.append(kps.extract_kp(sdf, sample_i, iterations=3))
        self.kps = np.stack(keypoints)
        self.vertices = np.array([
            [0, 0],
            [self.w-1, 0],
            [self.w-1, self.h-1],
            [0, self.h-1]
        ]).T

    def __len__(self):
        return self.N