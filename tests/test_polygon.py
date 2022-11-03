import polygon
import numpy as np


def test_area():
    verts = np.array([
        [1., 1, -1, -1],
        [1., -1, -1, 1]
    ])

    assert polygon.polygonArea(verts) == 4.0


def test_conversions():
    verts = np.array([
        [1., 1, -1, -1],
        [1., -1, -1, 1],
        [1., 1., 1., 1]
    ])

    pygame_verts = polygon.to_pygame_poly(verts)
    final_verts = polygon.from_pygame_poly(pygame_verts)
    assert np.allclose(final_verts, verts)
