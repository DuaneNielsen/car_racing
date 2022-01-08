from lshash import lshash
import numpy as np
import keypoints as kps
import geometry as geo
from matplotlib import pyplot as plt


class Episode:
    def __init__(self, i):
        self.sdfs = np.load(f'data/ep{i}_sdf_road.npy')[70:]
        self.states = np.load(f'data/ep{i}_state.npy')[70:]
        self.N, self.h, self.w = self.sdfs.shape
        sample_i = geo.grid_sample(self.h, self.w, 12, pad=4)
        keypoints = []
        for i, sdf in enumerate(self.sdfs):
            keypoints.append(kps.extract_kp(sdf, sample_i, iterations=3))
        self.kps = np.stack(keypoints)

    def __len__(self):
        return self.N


memory_set, query_set = [Episode(5), Episode(6), Episode(7)], Episode(2)


def knn(query, keypoints, num_results=3):
    distance = keypoints - query.reshape(1, *query.shape)
    distance = distance ** 2
    distance = distance.sum(axis=1)
    distance = np.sqrt(distance)
    distance = distance.mean(axis=1)
    return np.argsort(distance)[0:num_results]


fig = plt.figure()
axes = fig.subplots(2, 4)
query_plt = axes[0, 0]
results_ax = axes[0, 1:4]
query_state_plt = axes[1, 0]
results_state_ax = axes[1, 1:4]

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

    for memory, result_ax, result_state_ax in zip(memory_set, results_ax, results_state_ax):
        top_k = knn(kp_q, memory.kps, num_results=1)
        j = top_k[0]
        result_ax.imshow(memory.sdfs[j])
        result_state_ax.imshow(memory.states[j])
    plt.pause(0.01)
plt.show()