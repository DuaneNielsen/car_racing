# import the necessary packages
import numpy as np
import cv2
from matplotlib import pyplot as plt

if __name__ == '__main__':
    poses = np.load('data/ep0_pose.npy')
    sdfs = np.load('data/ep0_sdf.npy')
    states = np.load('data/ep0_state.npy')

    h, w = (500, 400)
    sdf_world = np.full((h, w), np.inf)
    state_world = np.full((h, w, 3), 255, dtype=np.uint8)
    offset = np.zeros((2, 3))
    offset[0, 2] = 100.0
    offset[1, 2] = 300.0

    fig = plt.figure()
    axes = fig.subplots(2, 2)
    sdf_plot, state_plot = axes[0]
    sdf_world_ax, state_world_ax = axes[1]
    update_freq = 10

    def update(ax, image, pose):
        ax.clear()
        warped = cv2.warpAffine(image, pose[0:2] + offset, (w, h))
        ax.imshow(warped)

    def update_world(sdf, state, pose):
        warped = cv2.warpAffine(sdf, pose[0:2] + offset, (w, h), borderValue=255.0, borderMode=0)
        warped_state = cv2.warpAffine(state, pose[0:2] + offset, (w, h), borderValue=255, borderMode=0)
        mask = warped == 0
        sdf_world[mask] = warped[mask]
        state_world[mask] = warped_state[mask]
        sdf_world_ax.imshow(sdf_world)
        state_world_ax.imshow(state_world)

    cnt = 0
    for pose, sdf, state in zip(poses, sdfs, states):
        update(sdf_plot, sdf, pose)
        update(state_plot, state[:60], pose)
        #update_world(sdf_world_ax, sdf, pose)
        update_world(sdf, state[:60], pose)

        if cnt % update_freq  == 0:
            plt.pause(0.05)
        cnt += 1

    plt.show()