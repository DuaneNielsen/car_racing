import numpy as np
import scipy.interpolate as interpolate
import matplotlib.pyplot as plt


"""
Minimal demo of using scipy to interpolate an image to a grid using a pose
"""

def M(t):
    return np.array([
        [np.cos(t[2]), -np.sin(t[2]), t[0]],
        [np.sin(t[2]), np.cos(t[2]), t[1]],
        [0., 0., 1.]
    ])


def invert_M(M):
    xy = np.matmul(-M[0:2, 0:2].T, M[0:2, 2])
    return np.array([
        [M[0, 0].item(), M[1, 0].item(), xy[0].item()],
        [M[0, 1].item(), M[1, 1].item(), xy[1].item()],
        [0., 0., 1.]
    ])


ep = 12
first_frame = 70
map_size = 200

sdfs = np.load(f'../data/ep{ep}_sdf.npy')[first_frame:].transpose(0, 2, 1)
poses = np.load(f'../data/ep{ep}_pose.npy')[first_frame:]

start_pose = invert_M(poses[0])
grid_x, grid_y = np.meshgrid(np.arange(map_size), np.arange(map_size))

fig = plt.figure()
gs = fig.add_gridspec(1, 3)
ax_nearest = fig.add_subplot(gs[0, 0])
ax_linear = fig.add_subplot(gs[0, 1])
ax_cubic = fig.add_subplot(gs[0, 2])
axes = [ax_nearest, ax_linear, ax_cubic]
fig.show()

for pose, sdf in zip(poses, sdfs):
    # center pose in map
    pose = np.matmul(start_pose, pose)
    pose = np.matmul(M(np.array([map_size / 2, map_size / 2, 0.])), pose)
    grid_0, grid_1 = np.meshgrid(*[np.arange(d) for d in sdf.shape])
    grid_0, grid_1 = grid_0.flatten(), grid_1.flatten()
    points = np.stack([grid_0, grid_1, np.ones_like(grid_0)])
    points = np.matmul(pose, points)[0:2]
    values = sdf[grid_0, grid_1]

    grid_z0 = interpolate.griddata(points.T, values, (grid_x, grid_y), method='nearest')
    grid_z1 = interpolate.griddata(points.T, values, (grid_x, grid_y), method='linear')
    grid_z2 = interpolate.griddata(points.T, values, (grid_x, grid_y), method='cubic')

    [ax.clear() for ax in axes]
    ax_nearest.imshow(grid_z0.T, extent=(0, 1, 0, 1), origin='lower')
    ax_linear.imshow(grid_z1.T, extent=(0, 1, 0, 1), origin='lower')
    ax_cubic.imshow(grid_z2.T, extent=(0, 1, 0, 1), origin='lower')
    fig.show()
    fig.canvas.draw()