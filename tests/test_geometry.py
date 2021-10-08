import numpy as np
from overlap import line_eqn_coeff, inside

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
    lines = line_eqn_coeff(tri)
    assert np.allclose(lines[:, 0], np.array([0, 1.0, 0]))
    assert np.allclose(lines[:, 1], np.array([-1., -1., 1]))
    assert np.allclose(lines[:, 2], np.array([1., 0.0, 0]))

    lines = line_eqn_coeff(rect)
    assert np.allclose(lines[:, 0], np.array([0, 1.0, 0]))
    assert np.allclose(lines[:, 1], np.array([-1, 0.0, 1]))
    assert np.allclose(lines[:, 2], np.array([0, -1.0, 1]))
    assert np.allclose(lines[:, 3], np.array([1., 0.0, 0]))


def test_inside():
    in_tri = inside(q, tri)
    assert np.all(~np.logical_xor(np.array([True, True, True, False, True, False, False, False]), in_tri))

    in_rect = inside(q, rect)
    assert np.all(~np.logical_xor(np.array([True, True, True, True, True, False, False, False]), in_rect))