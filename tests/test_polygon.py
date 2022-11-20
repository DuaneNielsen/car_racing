import polygon
import numpy as np
import torch


def test_area():
    verts = np.array([
        [1., 1, -1, -1],
        [1., -1, -1, 1]
    ])

    assert polygon.polygonArea(verts) == 4.0




def test_unflatten_index():
    index_start = torch.arange(3 * 4 * 5).reshape(3, 4, 5)

    co_ordinate = polygon.unflatten_index(index_start.flatten(), (3, 4, 5))
    index = index_start[co_ordinate]
    assert index_start.eq(index.unflatten(0, (3, 4, 5))).all()


def test_unflatten_index_partial():
    index_start = torch.arange(3 * 4 * 5).reshape(3, 4, 5)
    # shape = index_start.shape[:-1]
    # index_flat = index_start.flatten(end_dim=-2)
    # index = index.unflatten(0, shape)

    # print(index.shape)
    print(index_start[0, 0, 1])
    print(index_start[0, 1, 0])
    print(index_start[1, 0, 0])

    def unflatten_index(index, dim, shape):
        """
        takes a set of indices that were flattened and converts them to the unflattened values
        """
        denominator = 1
        co_ordinates = []
        for i in reversed(shape):
            co_ordinates += [(index // denominator) % i]
            denominator *= i

        for d in range(dim+1, len(index.shape) - dim):
            co_ordinates += [index.index_select(d, torch.arange(index.shape[d]))]

        return tuple(reversed(co_ordinates))

    co_ordinate = unflatten_index(index_start.flatten(end_dim=-2), 0, (3, 4))
    index = index_start[co_ordinate]

    assert index_start.eq(index.unflatten(0, (3, 4, 5))).all()


def test_se2_from_adjoint():

    def compare_angle(theta):
        se2 = torch.tensor([[0., 0., theta]])
        R = polygon.adjoint_matrix(se2)
        x = polygon.se2_from_adjoint(R)
        assert torch.allclose(se2 % (torch.pi * 2), x % (torch.pi * 2))

    compare_angle(torch.pi/2)
    compare_angle(torch.pi/4)
    compare_angle(torch.pi)
    compare_angle(-torch.pi / 4)
    compare_angle(torch.pi * 1.5)
    compare_angle(-torch.pi * 1.5)