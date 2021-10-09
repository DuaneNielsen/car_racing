import numpy as np
import geometry as geo
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import icp


if __name__ == '__main__':
    fig = plt.figure(figsize=(18, 10))
    axes = fig.subplots(1, 2)
    world, plt_error = axes
    world.set_aspect('equal')

    error = []

    world_frame = geo.Frame()

    while True:

        scan1 = geo.Scan(20, 30)
        scan2 = geo.Scan(20, 30)
        kp = np.array([
            [-2., 0],
            [0, -2],
            [1, 0],
            [0, 1]
        ]).T
        kp1_center = np.array([18, 11]).reshape(2, 1)
        kp2_center = np.array([22, 6]).reshape(2, 1)
        kp1 = kp + kp1_center
        kp2 = np.matmul(geo.R(np.radians(-90)), kp)
        kp2 = kp2 + kp2_center

        def draw_scan_in_world(scan, color):
            scan_v_wf = geo.transform_points(scan.M, scan.vertices.T)
            world.add_patch(Polygon(scan_v_wf.T, color=color, fill=False))
            world.autoscale_view()

        def draw_kp_in_world(kp, scan, color=None):
            P = geo.transform_points(scan.M, kp)
            world.scatter(P[0], P[1], color=color)

        def draw():
            [ax.clear() for ax in axes]
            draw_scan_in_world(scan1, color=[1, 0, 0])
            draw_scan_in_world(scan2, color=[0, 1, 0])
            draw_kp_in_world(kp1, scan1, color=[1, 0, 0])
            draw_kp_in_world(kp2, scan2, color=[0, 1, 0])
            draw_kp_in_world(kp2_centroid_w, world_frame)
            plt_error.plot(error)


        for i in range(2000):

            # compute all alignments in world space
            kp1_w = geo.transform_points(scan1.M, kp1)
            kp2_w = geo.transform_points(scan2.M, kp2)
            kp2_centroid_w = kp2_w.mean(axis=1, keepdims=True)
            e = icp.rms(kp2_w, kp1_w)
            error += [e]
            print(e)
            draw()
            plt.pause(1.0)

            R, t = icp.icp(kp2_w, kp1_w)
            scan2.t += t
            scan2.R = np.matmul(R, scan2.R)
