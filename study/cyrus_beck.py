import torch

"""
2D vertices are in (2, n) tensor
CLOCKWISE winding for polygons 
"""


def to_parametric(start, end):
    """
    parametric form of line_segmen given a start and end point

    [x, y ... ] = M t + B where t is in R

    returns a tuple (m: (2, N) vector for direction, b (2, N) vector for point on line)
    """
    m = end - start
    b = start
    return m, b


def rotate_verts(verts):
    """
    retuns a lise of vertices
    """
    range = torch.arange(1, verts.size(1) + 1)
    range[-1] = 0
    return verts.detach()[:, range]


def edges(verts):
    """
    edges of a polygon in form (start_vertex, end_vertex)
    """
    return verts, rotate_verts(verts)


def normal(start, end):
    """
    start and end points of line segment in 2D
    orthogonal vector to 2D line
    """
    m, b = to_parametric(start, end)
    return torch.stack((-m[1], m[0]))


def midpoint(start, end):
    """
    returns the co-ordinates of midpoint of a line segment
    """
    m, b = to_parametric(start, end)
    return m / 2 + b


if __name__ == '__main__':

    from matplotlib import pyplot as plt
    from matplotlib.patches import Polygon


    def clip_line_segment_by_poly(start, end, clipping_poly):
        """

        """
        with torch.no_grad():
            # Cyrus Beck algorithm
            # P0 - PEi
            # P0: start point of the line segment
            # P1: end point of the line segment
            # PEi: the vertices of the polygon
            _, M = start.shape
            _, N = clipping_poly.shape
            start = start.T.unsqueeze(1)
            end = end.T.unsqueeze(1)
            clipping_poly = clipping_poly.T

            start_minus_tri = start - clipping_poly
            end_minus_start = end - start

            # Ni -> compute outward facing Normals of the edges of the polygon
            edge_normals = normal(*edges(clipping_poly.T)).T

            # Ni dot (P0 - PEi)
            start_minus_verts_dot = (edge_normals * start_minus_tri).sum(-1)  # dot product if orthonormal basis!

            # Ni dot (P1 - P0)
            end_minus_start_dot = - (edge_normals * end_minus_start).sum(-1)  # dot product
            end_minus_start_dot[end_minus_start_dot == 0.] = -1.  # ignore zero denominator

            # t_values = Ni dot (P0 - PEi) / - Ni dot (P1 - P0)
            t_value = start_minus_verts_dot / end_minus_start_dot
            t_positive = t_value.clone()
            t_negative = t_value.clone()

            # min value of greater is 0, max value of lesser is 1.
            t_positive[~end_minus_start_dot.ge(0.)] = 0.
            t_negative[~end_minus_start_dot.lt(0.)] = 1.

            t_enter, _ = torch.max(t_positive, dim=1)
            t_exit, _ = torch.min(t_negative, dim=1)

            inside = ~t_enter.gt(t_exit)

            m, b = to_parametric(start.squeeze(1).T, end.squeeze(1).T)
            p_enter = m * t_enter + b
            p_exit = m * t_exit + b

            return torch.stack((p_enter.T, p_exit.T), dim=1), inside
            # segments = torch.stack((start.squeeze().T, p_g, p_l, end.squeeze().T))
            # return segments


    # plotting
    fig, axes = plt.subplots(2, 1)
    axes[0].set_aspect('equal', adjustable='box')
    axes[1].set_aspect('equal', adjustable='box')


    def plot_normals(ax, poly):
        midpoints = midpoint(*edges(poly))
        normals = normal(*edges(poly))
        mid_normal_ends = normals * 0.3 + midpoints
        for i in range(midpoints.size(1)):
            line_segment = torch.stack((midpoints[:, i], mid_normal_ends[:, i]), dim=1)
            ax.plot(line_segment[0], line_segment[1], color='green')


    def plot_line_segments(ax, segments, inside):
        color = ['blue', 'red', 'blue']
        for i in range(len(segments)):
            if inside[i]:
                ax.plot(segments[i, :, 0], segments[i, :, 1], color='red')
                print(segments[i, :, :])


    # define triangle and line segs
    start = torch.tensor([
        [0, 0, 0, 0],
        [1, 1, 1, 0.5]
    ])

    end = torch.tensor([
        [1, 1.0, 0., 1],
        [3, 1.5, 2., 1.1]
    ])

    triangle = torch.tensor([
        [0.5, 0.75, 0.5],
        [2.5, 1.75, 1.]
    ])

    # plot triangle
    axes[0].add_patch(Polygon(triangle.T))
    # plot_normals(axes[0], triangle)
    segments, inside = clip_line_segment_by_poly(start, end, triangle)
    plot_line_segments(axes[0], segments, inside)

    # define quad and line segs
    start = torch.tensor([
        [4.],
        [-1.]
    ])

    end = torch.tensor([
        [1.],
        [-3.]
    ])

    quad = torch.tensor([
        [3., 3., 2., 2.],
        [-1.5, -2.75, -2.75, -1.5]
    ])

    # plot quad
    axes[1].add_patch(Polygon(quad.T))
    # plot_normals(axes[1], quad)
    segments, inside = clip_line_segment_by_poly(start, end, quad)
    plot_line_segments(axes[1], segments, inside)

    plt.show()
