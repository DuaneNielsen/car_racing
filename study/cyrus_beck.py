import torch

"""
2D vertices are in (2, n) tensor
CLOCKWISE winding for polygons 
"""


def to_parametric(start, end):
    """
    parametric form of line_segment in D space

    start: (..., D) tensor of start points
    end: (..., D) tensor of end points

    returns a tuple (m: (..., D) vector for direction, b (..., D) vector for point on line)
    """
    m = end - start
    b = start
    return m, b


def rotate_verts(verts):
    """
    polygons: (P, V, 2) tensor of P polygons with V verts in CLOCKWISE winding
    returns a lise of vertices
    """
    range = torch.arange(1, verts.size(1) + 1)
    range[-1] = 0
    return verts.detach()[:, range]


def edges(polygons):
    """
    polygons: (P, V, 2) tensor of P polygons with V verts in CLOCKWISE winding
    returns tuple ( (P, V, 2) - start vertices, (P, V, 2) - end_vertices)
    """
    return polygons, rotate_verts(polygons)


def normal(start, end):
    """
    Computes the normal to a line segment
    start: (..., 2) set of start points in 2D
    end: (..., 2) set of end points in 2D
    returns: (..., 2) normals
    """
    m, b = to_parametric(start, end)
    return torch.stack((-m[..., 1], m[..., 0]), dim=-1)


def midpoint(start, end):
    """
    returns the co-ordinates of midpoint of a line segment
    """
    m, b = to_parametric(start, end)
    return m * 0.5 + b


if __name__ == '__main__':

    from matplotlib import pyplot as plt
    from matplotlib.patches import Polygon


    def clip_line_segment_by_poly(start, end, clipping_poly):
        """
        start: (N, 2) tensor containing N start points
        end: (N, 2) tensor containing N end points
        clipping_polygons: (P, V, 2) tensor containing P polygons of V vertices with CLOCKWISE winding
        returns:
            (N, 2) tensor of clip enter points
            (N, 2) tensor of clip exit points
            (N) boolean tensor, True if line intersects polygon
        """
        with torch.no_grad():
            # Cyrus Beck algorithm
            # P0 - PEi
            # P0: start point of the line segment
            # P1: end point of the line segment
            # PEi: the vertices of the polygon

            start = start.unsqueeze(1).unsqueeze(1)
            end = end.unsqueeze(1).unsqueeze(1)

            start_minus_tri = start - clipping_poly
            end_minus_start = end - start

            # Ni -> compute outward facing Normals of the edges of the polygon
            edge_normals = normal(*edges(clipping_poly))

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

            t_enter, _ = torch.max(t_positive, dim=2, keepdim=True)
            t_exit, _ = torch.min(t_negative, dim=2, keepdim=True)

            inside = ~t_enter.gt(t_exit)

            m, b = to_parametric(start.squeeze(2), end.squeeze(2))
            p_enter = m * t_enter + b
            p_exit = m * t_exit + b

            return p_enter, p_exit, inside


    # plotting
    fig, axes = plt.subplots(2, 1)
    axes[0].set_aspect('equal', adjustable='box')
    axes[1].set_aspect('equal', adjustable='box')


    def plot_normals(ax, poly):
        P, V, _ = poly.shape
        start, end = edges(poly)
        midpoints = midpoint(start, end)
        normals = normal(start, end)
        mid_normal_ends = normals * 0.3 + midpoints

        for p in range(P):
            for i in range(V):
                line_segment = torch.stack((midpoints[p, i], mid_normal_ends[p, i]), dim=0)
                ax.plot(line_segment[:, 0], line_segment[:, 1], color='green')


    def plot_line_segments(ax, start, end, mask=None, color=None):
        for i in range(len(start)):
            if mask is None:
                ax.plot([start[i, 0], end[i, 0]], [start[i, 1], end[i, 1]], color=color)
            elif mask[i]:
                ax.plot([start[i, 0], end[i, 0]], [start[i, 1], end[i, 1]], color=color)

    # define triangle and line segs
    start = torch.tensor([
        [0, 0, 0, 0.0, 0.6, 0.75, 0., 0.],
        [1, 1, 1, 0.5, 1.8, 1.25, 1., 1.6]
    ]).T

    end = torch.tensor([
        [1, 1.0, 0., 1.0, 1., 0.6, 1.0, 2.],
        [3, 1.5, 2., 1.1, 2., 1.5, 1.0, 1.6]
    ]).T

    triangles = []
    triangles += [torch.tensor([
        [0.5, 0.75, 0.5],
        [2.5, 1.75, 1.]
    ]).T]
    triangles += [torch.tensor([
        [1.5, 1.75, 1.5],
        [2.5, 1.75, 1.]
    ]).T]

    triangles = torch.stack(triangles)

    # plot triangle
    for triangle in triangles:
        axes[0].add_patch(Polygon(triangle))

    # plot_normals(axes[0], triangles)
    plot_line_segments(axes[0], start, end, color='blue')
    start, end, inside = clip_line_segment_by_poly(start, end, triangles)
    plot_line_segments(axes[0], start.flatten(0, 1), end.flatten(0, 1), inside.flatten(0, 1), color='red')

    # define quad and line segs
    start = torch.tensor([
        [+4., +2.5, +1.5],
        [-1., -2.5, -3.0]
    ]).T

    end = torch.tensor([
        [+1., +3.5, +3.5],
        [-3., -2.0, -3.0]
    ]).T

    quads = [torch.tensor([
        [3., 3., 2., 2.],
        [-1.5, -2.75, -2.75, -1.5]
    ]).T]
    quads += [torch.tensor([
        [+4., +4, +3.5, +3],
        [-1., -3, -3.0, -1]
    ]).T]

    quads = torch.stack(quads)

    # plot quad
    for quad in quads:
        axes[1].add_patch(Polygon(quad))
    # plot_normals(axes[1], quad)
    plot_line_segments(axes[1], start, end, color='blue')
    start, end, inside = clip_line_segment_by_poly(start, end, quads)
    plot_line_segments(axes[1], start.flatten(0, 1), end.flatten(0, 1), inside.flatten(0, 1), color='red')

    plt.show()
