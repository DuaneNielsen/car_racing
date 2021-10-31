import numpy as np
import keypoints as kps
import geometry as geo
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import cv2

class Episode:
    def __init__(self, i):
        self.sdfs = np.load(f'data/ep{i}_sdf_road.npy')[70:]
        self.states = np.load(f'data/ep{i}_state.npy')[70:]
        self.pose = np.load(f'data/ep{i}_pose.npy')[70:]
        self.pose_d = np.load(f'data/ep{i}_pose_d.npy')[70:]
        self.map = np.load(f'data/ep{i}_map.npy')
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


memory_set, query_set = [Episode(5), Episode(6), Episode(7)], Episode(0)


def knn(query, keypoints, num_results=3):
    distance = keypoints - query.reshape(1, *query.shape)
    distance = distance ** 2
    distance = distance.sum(axis=1)
    distance = np.sqrt(distance)
    distance = distance.mean(axis=1)
    return np.argsort(distance)[0:num_results]


def make_grid(h, w, homo=False):
    axis = []
    axis += [*np.meshgrid(np.arange(h), np.arange(w))]
    if homo:
        axis += [np.ones_like(axis[0])]
    return np.stack(axis, axis=2)


fig = plt.figure()
axes = fig.subplots(3, 4)
query_plt = axes[0, 0]
results_ax = axes[0, 1:4]
query_state_plt = axes[1, 0]
results_state_ax = axes[1, 1:4]
query_map_ax = axes[2, 0]
results_map_ax = axes[2, 1:4]

# fig2 = plt.figure()
# world_ax = fig2.subplots()
#
#
# def draw_box(ax, episode, j, color='blue'):
#     world_origin = np.array([episode.map.shape[0] // 2, episode.map.shape[1] // 2])
#     verts_w = geo.transform_points(memory.pose[j], episode.vertices)[0:2] + world_origin.reshape(2, 1)
#     ax.clear()
#     ax.imshow(episode.map)
#     ax.add_patch(Polygon(verts_w.T, color=color, fill=False))


skip = 0
for i in range(len(query_set)):
    skip += 1
    if skip % 3 != 0:
        continue
    for row in axes:
        for ax in row:
            ax.clear()
    query_plt.imshow(query_set.sdfs[i])
    query_state_plt.imshow(query_set.states[i])
    kp_q = query_set.kps[i]

    for memory, result_ax, result_state_ax, result_map_ax in zip(memory_set, results_ax, results_state_ax, results_map_ax):
        top_k = knn(kp_q, memory.kps, num_results=1)
        j = top_k[0]
        result_ax.clear()
        result_ax.imshow(memory.sdfs[j])
        result_state_ax.clear()
        result_state_ax.imshow(memory.states[j])

        world = np.full((1000, 1000), 4.0)
        world_n = np.zeros_like(world)
        world_origin = np.array([world.shape[0] // 2, world.shape[1] // 2])
        max_h, min_h, max_w, min_w = 0, world.shape[1], 0, world.shape[0]

        # initialize segment with the correct rotation
        delta = memory.pose[j]
        delta[2, 0:1] = 0.0
        initial_r = delta

        for k in range(j, j+90):
            # write to world array
            if k >= len(memory):
                break
            image = memory.sdfs[k].clip(-4.0, 4.0)
            mask = (image > -4.0) & (image < 4.0)

            model_grid = make_grid(memory.w, memory.h, homo=True).reshape(memory.h * memory.w, 3).T
            model_grid = model_grid[:, mask.reshape(memory.h * memory.w)]

            #model_grid_w = np.matmul(memory.pose[k], model_grid)[0:2] + world_origin.reshape(2, 1)
            model_grid_w = np.matmul(delta, model_grid)[0:2] + world_origin.reshape(2, 1)
            model_grid_w = model_grid_w.round().astype(np.int64)
            h_i, w_i = model_grid_w[0], model_grid_w[1]

            n = world_n[w_i, h_i]
            world[w_i, h_i] = ((world[w_i, h_i] * n + image[model_grid[1], model_grid[0]]) / (n + 1))
            world_n[w_i, h_i] += 1.
            max_h, min_h = max(h_i.max(), max_h), min(h_i.min(), min_h)
            max_w, min_w = max(w_i.max(), max_w), min(w_i.min(), min_w)

            delta = np.matmul(memory.pose_d[k], delta)
            # result_map_ax.imshow(world[min_w:max_w, min_h:max_h], cmap='cool')
            # draw_box(world_ax, memory, k)
            # plt.pause(0.05)

        result_map_ax.clear()
        result_map_ax.imshow(world[min_w:max_w, min_h:max_h], cmap='cool')

    plt.pause(0.01)
plt.show()