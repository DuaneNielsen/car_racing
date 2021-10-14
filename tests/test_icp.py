import numpy as np
import icp
import geometry as geo
from matplotlib import pyplot as plt

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
    f1.theta = np.radians(1.0)
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

    def generate_trajectory(self, dx, dy, dtheta):

        gt_frames = []

        for i in range(4):
            f0 = geo.Frame()
            f0.x, f0.y, f0.theta = self.x, self.y, self.theta
            gt_frames += [f0]
            dt = np.matmul(f0.R, np.array([dx, dy]).reshape(2, 1))
            self.x += dt[0]
            self.y += dt[1]
            self.theta += dtheta

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
