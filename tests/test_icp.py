import numpy as np
import icp

kp = np.array([
    [-1., 0],
    [0, -1],
    [1, 0],
    [0, 1]
]).T


def test_icp_translate():
    t = np.array([+5.0, +7.0]).reshape(2, 1)
    kp2 = kp + t
    R, t_ = icp.icp(kp2, kp)
    assert np.allclose(t, -t_)