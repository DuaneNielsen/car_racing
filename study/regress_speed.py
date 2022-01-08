import numpy as np
from matplotlib import pyplot as plt
import geometry as geo

if __name__ == '__main__':
    ep = 0
    pose = np.load(f'data/ep{0}_pose.npy')
    sdf = np.load(f'data/ep{0}_sdf.npy')
    state = np.load(f'data/ep{0}_state.npy')
    gt = np.load(f'data/ep{0}_gt.npy')

    origin = np.zeros((2, 1))

    d_pose = []
    gt_speed = []

    for t in range(len(pose)-1):
        d_pose += [geo.transform_points(pose[t+1], origin) - geo.transform_points(pose[t], origin)]
        gt_speed += [gt[t + 1] - gt[t]]
    d_pose = np.stack(d_pose)
    gt_speed = np.stack(gt_speed)

    gt_speed *= 3.0

    fig = plt.figure()
    pos, speed = fig.subplots(1, 2)
    fig.show()

    pos.scatter(pose[:, 1, 2], pose[:, 0, 2])
    pos.scatter(gt[:, 0], gt[:, 1])

    speed.scatter(d_pose[:, 0], d_pose[:, 1])
    speed.scatter(gt_speed[:, 0], gt_speed[:, 1])
    plt.show()
