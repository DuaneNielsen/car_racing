import numpy as np
import icp

kp = np.array([
    [-1., 0],
    [0, -1],
    [1, 0],
    [0, 1]
]).T

range = np.arange(10)
source_pts = np.stack([range, range + 10])
target_pts= np.stack([range, range + 10])


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