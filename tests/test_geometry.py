import numpy as np
import geometry as geo


tri = np.array([
    [0, 0],
    [1, 0],
    [0, 1]
]).T

rect = np.array([
    [0, 0],
    [1, 0],
    [1, 1],
    [0, 1]
]).T

q = np.array([
    [0., 0],
    [0, 1],
    [1, 0],
    [1, 1],
    [0.5, 0.5],
    [-1, -1],
    [4, 4.0],
    [-0.5, 0.5]
]).T


def test_line_coeff():
    lines = geo.line_eqn_coeff(tri)
    assert np.allclose(lines[:, 0], np.array([0, 1.0, 0]))
    assert np.allclose(lines[:, 1], np.array([-1., -1., 1]))
    assert np.allclose(lines[:, 2], np.array([1., 0.0, 0]))

    lines = geo.line_eqn_coeff(rect)
    assert np.allclose(lines[:, 0], np.array([0, 1.0, 0]))
    assert np.allclose(lines[:, 1], np.array([-1, 0.0, 1]))
    assert np.allclose(lines[:, 2], np.array([0, -1.0, 1]))
    assert np.allclose(lines[:, 3], np.array([1., 0.0, 0]))


def test_inside():
    in_tri = geo.inside(q, tri)
    assert np.all(~np.logical_xor(np.array([True, True, True, False, True, False, False, False]), in_tri))

    in_rect = geo.inside(q, rect)
    assert np.all(~np.logical_xor(np.array([True, True, True, True, True, False, False, False]), in_rect))


def test_inverse():
    f1 = geo.Scan()
    assert np.allclose(np.matmul(f1.M, f1.inv_M), np.eye(3))

    f2 = geo.Scan(x=5, y=8, theta=np.radians(30))
    assert np.allclose(np.matmul(f2.inv_M, f2.M), np.eye(3))


def test_homo():
    theta = np.radians(30)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    x, y = 2.0, 3.0
    t = np.array([x, y]).reshape(2, 1)

    def homoM(R, t):
        M = np.concatenate((R, t), axis=1)
        return np.concatenate((M, np.array([[0., 0, 1]])), axis=0)

    tri_h = np.concatenate((tri, np.ones((1, tri.shape[1]))))
    tri_h = np.matmul(homoM(R, t), tri_h)[0:2]

    # rotate first then translate
    tri_r_t = np.matmul(R, tri) + t
    assert np.allclose(tri_h, tri_r_t)

    # translate first then rotate
    tri_t_r = np.matmul(R, tri + t)
    assert ~np.allclose(tri_h, tri_t_r)