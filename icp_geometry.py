import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon

import geometry as geo
import icp


def main():

    fig = plt.figure()
    axes = fig.subplots(2, 4)
    f0_plt, f1_plt, f2_plt, world_plt = (0, 0), (0, 1), (0, 2), (0, 3)
    gt_f0_plt, gt_f1_plt, gt_f2_plt, gt_world_plt = (1, 0), (1, 1), (1, 2), (1, 3)

    def t(x, y):
        return np.array([x, y]).reshape(2, 1)

    kp_w = np.array([
        [10, 10],
        [20, 10],
        [10, 20]
    ]).T

    h, w = 30, 40

    def gt_frame(x, y, theta, h=30, w=40):
        scan = geo.Scan(h, w, x=x, y=y, theta=theta)
        image = geo.transform_points(scan.inv_M, kp_w)
        scan.image = image
        return scan

    gt_f0 = gt_frame(0, 0, np.radians(0.0), h=h, w=w)
    gt_f1 = gt_frame(0, 7.0, np.radians(45.), h=h, w=w)
    gt_f2 = gt_frame(0, 14.0, np.radians(90.), h=h, w=w)

    def plot_scan(axis, f, color, R=None, t=None):
        axes[axis].add_patch(Polygon(f.vertices.T, color=color, fill=False))
        axes[axis].scatter(*np.split(f.image, 2))
        axes[axis].scatter(*np.split(f.image.mean(axis=1),2))

    plot_scan(gt_f0_plt, gt_f0, 'blue')
    plot_scan(gt_f1_plt, gt_f1, 'green')
    plot_scan(gt_f2_plt, gt_f2, 'red')

    def plot_world(world_plt, scans):

        colors = ['blue', 'green', 'red']

        for i, (scan, color) in enumerate(zip(scans, colors)):
            kp_w = geo.transform_points(scan.M, scan.image)
            patch = Polygon(geo.transform_points(scan.M, scan.vertices).T, color=color, fill=False)
            axes[world_plt].add_patch(patch)
            axes[world_plt].scatter(*np.split(kp_w, 2), label=f'f{i}', color=color)

    plot_world(gt_world_plt, [gt_f0, gt_f1, gt_f2])

    gt_frames = [gt_f0, gt_f1, gt_f2]
    est_frames = [geo.Scan(h, w, image=gt_f0.image)]
    est_T = []

    for step in range(1, 3):
        prev = est_frames[step-1]
        curr = geo.Scan(h, w, image=gt_frames[step].image, x=prev.x, y=prev.y, theta=prev.theta)

        R, t, kp_t0, kp_t1, best_error = icp.ransac_icp(source=curr.image, target=prev.image, k=1, n=3, seed=105)
        print(f'error: {best_error}, t: {t.squeeze()}, R: {np.degrees(geo.theta(R))} n: {kp_t0.shape[1]}')  # {t_rot}')

        # translate in the f1 reference frame as the above will return the relative transformation
        curr.t = curr.t + np.matmul(curr.R, t)
        curr.R = np.matmul(R, curr.R)


        est_frames.append(curr)
        est_T.append([R, t])

    plot_scan(f0_plt, est_frames[0], 'blue', *est_T[0])
    plot_scan(f1_plt, est_frames[1], 'green', *est_T[1])
    plot_scan(f2_plt, est_frames[2], 'red')

    plot_world(world_plt, est_frames)

    plt.legend()
    plt.show()

    # assert np.allclose(geo.transform_points(f0.M, kp_f0), geo.transform_points(f1.M, kp_f1))

if __name__ == '__main__':
    main()