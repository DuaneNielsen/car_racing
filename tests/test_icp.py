import numpy as np
import icp
import geometry as geo
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import copy
from numpy.linalg import norm


kp = np.array([
    [-1., 0],
    [0, -1],
    [1, 0],
    [0, 1]
]).T

r = np.arange(10)
source_pts = np.stack([r, r + 10])
target_pts= np.stack([r, r + 10])


def test_icp_translate():
    t = np.array([+5.0, +7.0]).reshape(2, 1)
    kp2 = kp + t
    R, t_ = icp.icp(kp2, kp)
    assert np.allclose(t, -t_)


def test_ransac_sample():
    source, target = icp.ransac_sample(source_pts, target_pts, 3, 5, seed=101)
    print(source[0])
    assert np.allclose(source[0], np.array(
        [[1, 6,  6,  1,  1],
         [11, 16, 16, 11, 11]]
    ))


def test_ransac_icp():
    icp.ransac_icp(source_pts, target_pts, 3, 5)


def test_icp_alignment():
    """
    Unit test that generates perfect key-points
    """

    f0 = geo.Frame()
    f1 = geo.Frame()
    f1.x = 1.0
    f1.y = 1.0
    f1.theta = np.radians(30.0)
    kp_source = geo.grid_sample(100, 100, 10, pad=10)
    kp_target = geo.transform_points(f1.M, kp_source)
    R, t, source, target, best_error = icp.ransac_icp(kp_source, kp_target, k=10, n=5)

    f0.t = f0.t + t
    f0.R = np.matmul(R, f0.R)

    print('')
    print(f0.M)
    print(f1.M)

    assert np.allclose(f0.M, f1.M)


class Generator():
    def __init__(self):
        self.x = 0.
        self.y = 0.
        self.theta = 0.

    def generate_trajectory(self, dx, dy, dtheta, length=2):

        gt_frames = [geo.Frame()]

        for i in range(1, length):
            f = copy.copy(gt_frames[i-1])
            f.R = np.matmul(geo.R(dtheta), f.R)
            dt = np.matmul(f.R, np.array([dx, dy]).reshape(2, 1))
            f.t = f.t + dt
            gt_frames += [f]

        return gt_frames


def kps(f0, f1):
    grid = geo.grid_sample(100, 100, 10, pad=10)
    return geo.transform_points(f0.M, grid), geo.transform_points(f1.M, grid)





def plot_odometry(gt_frames, icp_frames):
    fig = plt.figure()
    pos_plt, rot_plt, kp_plt = fig.subplots(3, 1)

    pos = np.concatenate([frame.t for frame in icp_frames], axis=1)
    pos_gt = np.concatenate([frame.t for frame in gt_frames], axis=1)
    pos_plt.scatter(pos_gt[0], pos_gt[1], color='blue', label='gt')
    pos_plt.scatter(pos[0], pos[1], color='red', label='estimate')

    rot = [np.degrees(frame.theta) for frame in icp_frames]
    rot_gt = [np.degrees(frame.theta) for frame in gt_frames]
    rot_plt.scatter(range(len(rot_gt)), rot_gt, color='blue', label='gt')
    rot_plt.scatter(range(len(rot)), rot, color='red', label='estimate')

    pos_plt.legend()
    rot_plt.legend()
    plt.pause(0.5)


def align(gt_frames):

    icp_frames = [geo.Frame()]
    for step in range(len(gt_frames) - 1):
        f0 = icp_frames[step]
        f1 = geo.Frame()
        f1.t = f0.t.copy()
        f1.R = f0.R.copy()
        icp_frames += [f1]

        kp_source, kp_target = kps(gt_frames[step], gt_frames[step + 1])

        R, t, kp_t0, kp_t1, best_error = icp.ransac_icp(kp_source, kp_target, k=10, n=5, seed=105)
        print(f'error: {best_error}, t: {t.squeeze()}, R: {np.degrees(geo.theta(R))} t_rot: ')#{t_rot}')

        # translate in the f1 reference frame as the above will return the relative transformation
        f1.t = f1.t + np.matmul(f1.R, t)
        f1.R = np.matmul(R, f1.R.copy())

        # kp_plt.clear()
        # kp_plt.scatter(kp_t0[0], kp_t0[1], color='blue')
        # kp_plt.scatter(kp_t1[0], kp_t1[1], color='red')

    return icp_frames


def test_icp_alignment_x():
    """
    Unit test that generates perfect key-points
    """

    dx, dy, dtheta = 1.0, 0., np.radians(0.0)
    gt_frames = Generator().generate_trajectory(dx, dy, dtheta)
    icp_frames = align(gt_frames)
    plot_odometry(gt_frames, icp_frames)
    for gt, icp in zip(gt_frames, icp_frames):
        assert np.allclose(gt.t, icp.t)
        assert np.allclose(gt.R, icp.R)


def test_icp_alignment_y():
    """
    Unit test that generates perfect key-points
    """

    dx, dy, dtheta = 0., 1., np.radians(0.0)
    gt_frames = Generator().generate_trajectory(dx, dy, dtheta)
    icp_frames = align(gt_frames)
    plot_odometry(gt_frames, icp_frames)
    for gt, icp in zip(gt_frames, icp_frames):
        assert np.allclose(gt.t, icp.t)
        assert np.allclose(gt.R, icp.R)


def test_icp_alignment_xy():
    """
    Unit test that generates perfect key-points
    """

    dx, dy, dtheta = 1., 1., np.radians(0.0)
    gt_frames = Generator().generate_trajectory(dx, dy, dtheta)
    icp_frames = align(gt_frames)
    plot_odometry(gt_frames, icp_frames)
    for gt, icp in zip(gt_frames, icp_frames):
        assert np.allclose(gt.t, icp.t)
        assert np.allclose(gt.R, icp.R)


def test_icp_alignment_R():
    """
    Unit test that generates perfect key-points
    """

    dx, dy, dtheta = 0., 0., np.radians(-45.0)
    gt_frames = Generator().generate_trajectory(dx, dy, dtheta)
    icp_frames = align(gt_frames)
    plot_odometry(gt_frames, icp_frames)
    for gt, icp in zip(gt_frames, icp_frames):
        assert np.allclose(gt.t, icp.t)
        assert np.allclose(gt.R, icp.R)


def test_icp_alignment_xR():
    """
    Unit test that generates perfect key-points
    """

    dx, dy, dtheta = 1., 0., np.radians(45.0)
    gt_frames = Generator().generate_trajectory(dx, dy, dtheta)
    icp_frames = align(gt_frames)
    plot_odometry(gt_frames, icp_frames)
    for gt, icp in zip(gt_frames, icp_frames):
        assert np.allclose(gt.t, icp.t)
        assert np.allclose(gt.R, icp.R)


def plot_frames(frames):

    for i in range(1, len(frames)):
        f0, f1 = frames[i-1], frames[i]
        plt.plot([f0.t[0], f1.t[0]],  [f0.t[1], f1.t[1]])
        l = norm(f1.t - f0.t)
        exp_t = np.matmul(f1.R, np.array([0, l]).reshape(2, 1)) + f0.t
        plt.plot([f0.t[0], exp_t[0]], [f0.t[1], exp_t[1]])

    plt.show()

def test_track():
    gt_track = Generator().generate_trajectory(0.0, 0.5, np.radians(5.0), length=3)
    icp_track = align(gt_track)
    plot_frames(gt_track)
    plot_odometry(gt_track, icp_track)
    plt.show()


def test_rotation_direction():

    def xy_axis(theta):
        return np.array([
            [np.cos(theta), -np.sin(theta)],
            [0, 0],
            [np.sin(theta), np.cos(theta)]
        ]).T

    source = xy_axis(0.0)
    target = xy_axis(np.radians(50.0))

    fig = plt.figure()
    fig.show()
    ax1, ax2 = fig.subplots(2)
    ax1.scatter(*np.split(source, 2))
    ax2.scatter(*np.split(target, 2))

    R, t = icp.icp(source, target)
    print(np.degrees(np.arccos(R[0, 0])))

    plt.show()




def test_frames():

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