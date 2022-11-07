import torch
from torch import cos, sin

"""
Line segs are tuples of (N, 2) start, (N, 2) end tensors
Polygons are in (P, V, 2) tensor -> P - polygon, V - vertices, 2 - dimensions
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


def roll(t, dim=0):
    """
    moves values in tensor one index to the right
    t: the tensor to roll
    dim: the dim to roll on
    """
    prev = torch.arange(1, t.size(dim) + 1)
    prev[-1] = 0
    return t.index_select(dim, prev)


def edges(polygons):
    """
    polygons: (P, V, 2) or (V, 2) tensor of P polygons with V verts in CLOCKWISE winding
    returns tuple ( (P, V, 2) or (V,2) - start vertices, (P, V, 2) or (V, 2) - end_vertices)
    """
    if len(polygons.shape) == 2:
        return polygons, roll(polygons, dim=0)
    else:
        return polygons, roll(polygons, dim=1)


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

        return p_enter, p_exit, inside.squeeze(-1)


def is_inside(es, ee, p):
    """
    checks if a point is inside an edge of a polygon with CLOCKWISE winding
    es: (..., 2) edge start co-ordinate
    ee: (..., 2) edge end co-ordinate
    p: (..., 2) point to test
    """
    R = (ee[..., 0] - es[..., 0]) * (p[..., 1] - es[..., 1]) - (ee[..., 1] - es[..., 1]) * (p[..., 0] - es[..., 0])
    return R <= 0


def compute_intersection(p1, p2, p3, p4):
    """
    given points p1 and p2 on line L1, compute the equation of L1 in the
    format of y = m1 * x + b1. Also, given points p3 and p4 on line L2,
    compute the equation of L2 in the format of y = m2 * x + b2.

    To compute the point of intersection of the two lines, equate
    the two line equations together

    m1 * x + b1 = m2 * x + b2

    and solve for x. Once x is obtained, substitute it into one of the
    equations to obtain the value of y.

    if one of the lines is vertical, then the x-coordinate of the point of
    intersection will be the x-coordinate of the vertical line. Note that
    there is no need to check if both lines are vertical (parallel), since
    this function is only called if we know that the lines intersect.
    """

    # if first line is vertical
    if p2[0] - p1[0] == 0:
        x = p1[0]

        # slope and intercept of second line
        m2 = (p4[1] - p3[1]) / (p4[0] - p3[0])
        b2 = p3[1] - m2 * p3[0]

        # y-coordinate of intersection
        y = m2 * x + b2

    # if second line is vertical
    elif p4[0] - p3[0] == 0:
        x = p3[0]

        # slope and intercept of first line
        m1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b1 = p1[1] - m1 * p1[0]

        # y-coordinate of intersection
        y = m1 * x + b1

    # if neither line is vertical
    else:
        m1 = (p2[1] - p1[1]) / (p2[0] - p1[0])
        b1 = p1[1] - m1 * p1[0]

        # slope and intercept of second line
        m2 = (p4[1] - p3[1]) / (p4[0] - p3[0])
        b2 = p3[1] - m2 * p3[0]

        # x-coordinate of intersection
        x = (b2 - b1) / (m1 - m2)

        # y-coordinate of intersection
        y = m1 * x + b1

    # need to unsqueeze so torch.cat doesn't complain outside func
    intersection = torch.stack((x, y)).unsqueeze(0)

    return intersection


def polygon_clip(subject_polygon, clipping_polygon):
    # it is assumed that requires_grad = True only for clipping_polygon
    # subject_polygon and clipping_polygon are P, V x 2 and M x 2 torch
    # tensors respectively

    final_polygon = torch.clone(subject_polygon)

    for i in range(len(clipping_polygon)):

        # stores the vertices of the next iteration of the clipping procedure
        # final_polygon consists of list of 1 x 2 tensors
        next_polygon = torch.clone(final_polygon)

        # stores the vertices of the final clipped polygon. This will be
        # a K x 2 tensor, so need to initialize shape to match this
        final_polygon = torch.empty((0, 2))

        # these two vertices define a line segment (edge) in the clipping
        # polygon. It is assumed that indices wrap around, such that if
        # i = 0, then i - 1 = M.
        c_edge_start = clipping_polygon[i - 1]
        c_edge_end = clipping_polygon[i]

        for j in range(len(next_polygon)):

            # these two vertices define a line segment (edge) in the subject
            # polygon
            s_edge_start = next_polygon[j - 1]
            s_edge_end = next_polygon[j]

            if is_inside(c_edge_start, c_edge_end, s_edge_end):
                if not is_inside(c_edge_start, c_edge_end, s_edge_start):
                    intersection = compute_intersection(s_edge_start, s_edge_end, c_edge_start, c_edge_end)
                    final_polygon = torch.cat((final_polygon, intersection), dim=0)
                final_polygon = torch.cat((final_polygon, s_edge_end.unsqueeze(0)), dim=0)
            elif is_inside(c_edge_start, c_edge_end, s_edge_start):
                intersection = compute_intersection(s_edge_start, s_edge_end, c_edge_start, c_edge_end)
                final_polygon = torch.cat((final_polygon, intersection), dim=0)

    return final_polygon


class Vector2:
    def __init__(self, x=0., y=0.):
        self.v = torch.tensor([[x, y]])

    @property
    def x(self):
        return self.v[0, 0]

    @property
    def y(self):
        return self.v[0, 1]

    @property
    def pygame(self):
        return self.v[0].tolist()

    def __add__(self, vec2):
        result = Vector2()
        result.v = self.v + vec2.v
        return result

    @property
    def homo(self):
        return torch.cat((self.v.T, torch.ones(1, 1)), dim=0)

    def rotate(self, theta):
        self.v = rotate2D(self.homo, theta)[0:2].T
        return self


# TODO fix area

# (X[i], Y[i]) are coordinates of i'th point.
def polygonArea(verts):
    # Initialize area
    area = 0.0
    n = verts.shape[1]

    # Calculate value of shoelace formula
    j = n - 1
    for i in range(0, n):
        area += (verts[0][j] + verts[0][i]) * (verts[1][j] - verts[1][i])
        j = i  # j is previous vertex to i

    # Return absolute value
    return int(abs(area / 2.0))


def rotate2D(verts, theta):
    theta = theta if isinstance(theta, torch.Tensor) else torch.tensor(theta)
    R = torch.tensor([
        [+cos(theta), sin(theta), 0],
        [-sin(theta), cos(theta), 0],
        [0., 0., 1.]
    ])
    return torch.matmul(R, verts)


def translate2D(verts, vec2):
    T = torch.tensor([
        [1., 0, vec2.x],
        [0., 1., vec2.y],
        [0., 0., 1.]
    ])
    return torch.matmul(T, verts)


def scale2D(verts, vec2):
    S = torch.tensor([
        [vec2.x, 0, 0.],
        [0., vec2.y, 0],
        [0., 0., 1.]
    ])
    return torch.matmul(S, verts)


def to_pygame_poly(verts):
    return [v for v in zip(verts[0], verts[1])]


class Polygon:
    def __init__(self, verts, pos=None, scale=None, theta=None):
        """
        Use CLOCKWISE winding for vertices

        x4, y4 -----------  x1, y1
          -                   -
          -                   -
          -                   -
          -                   -
        x3, y3 ------------ x2, y2

        square = Polygon([
          [x1, x2, x3, x4]
          [y1, y2, y3, y4]
        ])

        """
        self.verts = torch.tensor(verts).T
        self.scale = Vector2(1., 1.) if scale is None else scale
        self._theta = torch.tensor(0.) if theta is None else torch.tensor(theta)
        self.pos = Vector2(0., 0.) if pos is None else pos

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, theta):
        self._theta = torch.tensor(theta)

    @property
    def verts_homo(self):
        return torch.cat((self.verts, torch.ones(self.verts.shape[0], 1)), dim=-1).T

    @property
    def num_vertices(self):
        return self.verts.shape[0]

    @property
    def world_verts(self):
        M = translate2D(rotate2D(scale2D(torch.eye(3), self.scale), self._theta), self.pos)
        return torch.matmul(M, self.verts_homo).T[:, 0:2]

    @property
    def pygame_world_verts(self):
        return to_pygame_poly(self.world_verts.T)

    @staticmethod
    def stack(polys):
        """
        stack a list of P Polygons with V vertices into a P, V tensor
        polys: [Polygon(), Polygon()] must have the same number of vertices
        returns (P, V, 2) tensor of P polygons with V vertices
        """
        num_verts = [p.num_vertices for p in polys]
        for i in range(len(num_verts)):
            if num_verts[0] != num_verts[i]:
                raise Exception(f'polygon with index {i} had {num_verts[i]} vertices'
                                f' all polygons must have same number of vertices, eg: {num_verts[0]} vertices')
        return torch.stack([p.world_verts for p in polys])


def cross2D(v1, v2):
    """
    calculates the cross product in 2 dimensions
    v1: [..., 2] tensor
    v2: [..., 2] tensor
    """
    return v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]


def compute_tu(t_m, t_b, u_m, u_b):
    """
    calculates the point of intersection between two lines in parametric form
    t_m: (N, 2) m values: (unnormalized slope) for a set of N lines
    t_b: (N, 2) b values: (point on the line) for N lines
    u_m: (M, 2) m values: (unnormalized slope) for a set of N lines to intesect with t
    u_b: (M, 2) b values: (poin on the line) for a set of N lines to intesect with t
    returns t, u, t_m, t_b, U_m, u_b
        (M, N, 1) t: parameter such that t_b + t_m * t = intersection point (point in terms of t)
        (M, N, 1) u: parameter such that u_b + u_m * u = intersection point (point in terms of u)
        (1, M, 2) t_m: tm such that t_m * t + t_b = point of intersection
        (1, M, 2) t_b: see above
        (M, 1, 2) u_m: tm such that u_m * u + u_b = point of intersection
        (M, 1, 2) u_b: see above

        Note: t and u have the property that 0. < t < 1., and 0 < u < 1. means the point lies inside the line segment
        used to obtain the parametric form

        ie: 0. =< t =< 1. and 0. =< u =< 1 means the point lies on the intersection of 2 line segments used to obtain
        the parametric form

        Note: if the lines and parallel, no point of intersection exists, so t and u will be set to inf
    """
    t_m, t_b = t_m.unsqueeze(0), t_b.unsqueeze(0)
    u_m, u_b = u_m.unsqueeze(1), u_b.unsqueeze(1)
    i = u_b - t_b
    det_denominator = cross2D(t_m, u_m)
    parallel = det_denominator == 0.  # take note of the parallel lines
    det_denominator[parallel] = 1.  # put a placeholder value to prevent div/0
    t = cross2D(i, u_m) / det_denominator
    u = cross2D(i, t_m) / det_denominator

    # if the lines were parallel, they converge at "infinity"
    t[parallel] = torch.inf
    u[parallel] = torch.inf
    return t.unsqueeze(-1), u.unsqueeze(-1), t_m, t_b, u_m, u_b


def line_seg_intersect(t_start, t_end, u_start, u_end):
    """
    Computes intersection between two sets of line segments
    t_start: (N, 2) start points of segments
    t_end: (N, 2) end points of set of line segments
    u_start: (M, 2) start points of segment set to intersect
    u_end: (M, 2) end points of segment set to intersections
    returns: p, mask
        p: (N, M, 2)
        mask: (N, M)
    """
    t_m, t_b = to_parametric(t_start, t_end)
    u_m, u_b = to_parametric(u_start, u_end)

    t, u,  t_m, t_b, u_m, u_b = compute_tu(t_m, t_b, u_m, u_b)
    intersect = t.ge(0.) & t.le(1.) & u.ge(0.) & u.le(1.)
    p = t_m * t + t_b
    return p, intersect.squeeze()


def raycast(ray_origin, ray_vector, l_start, l_end, max_len=None):
    """
    casts a ray against a set of line segments
    ray_origin: (N, 2) ray origins
    ray_vector: (N, 2) ray vectors
    l_start: (M, 2) line segment starts to cast onto
    l_end: (M, 2) line segment ends to cast onto
    returns:
        ray_origin: (N, 2)
        ray_end: (N, 2)
        mask: N - True if the ray hit a line segment, False if it didn't
    """
    t_m, t_b = to_parametric(l_start, l_end)
    ray_vector = ray_vector/torch.linalg.vector_norm(ray_vector, dim=-1, keepdim=True)
    t, r,  t_m, t_b, ray_vector, ray_origin = compute_tu(t_m, t_b, ray_vector, ray_origin)

    if max_len is None:
        intersect = t.ge(0.) & t.lt(1.0) & r.ge(0.)
        r[~intersect] = torch.inf
    else:
        intersect = t.ge(0.) & t.lt(1.0) & r.ge(0.) & r.lt(max_len)
        r[~intersect] = max_len

    r_length, r_index = torch.min(r, dim=1, keepdim=True)
    ray_end = ray_origin + r_length * ray_vector
    return ray_origin.squeeze(1), ray_end.squeeze(1), r_length.squeeze()


if __name__ == '__main__':

    from matplotlib import pyplot as plt
    from matplotlib.patches import Polygon as PolygonPatch

    # plotting
    fig, axes = plt.subplots(2, 3)
    axes = axes.flatten()

    for ax in axes:
        ax.set_aspect('equal', adjustable='box')


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


    def plot_line_segments(ax, start, end, mask=None, color=None, **kwargs):
        for i in range(len(start)):
            if mask is None:
                ax.plot([start[i, 0], end[i, 0]], [start[i, 1], end[i, 1]], color=color, **kwargs)
            elif mask[i]:
                ax.plot([start[i, 0], end[i, 0]], [start[i, 1], end[i, 1]], color=color, **kwargs)


    """
    top left subplot - lines clipping 2 triangles
    """

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
        axes[0].add_patch(PolygonPatch(triangle))

    # plot_normals(axes[0], triangles)
    plot_line_segments(axes[0], start, end, color='blue')
    start, end, inside = clip_line_segment_by_poly(start, end, triangles)
    plot_line_segments(axes[0], start.flatten(0, 1), end.flatten(0, 1), inside.flatten(0, 1), color='red')

    """
    top right diagram - lines clipping 2 quads
    """

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
        axes[1].add_patch(PolygonPatch(quad))
    # plot_normals(axes[1], quad)
    plot_line_segments(axes[1], start, end, color='blue')
    start, end, inside = clip_line_segment_by_poly(start, end, quads)
    plot_line_segments(axes[1], start.flatten(0, 1), end.flatten(0, 1), inside.flatten(0, 1), color='red')

    """
    bottom left, polygon clipping with a polygon
    """

    quads = [
        Polygon([
            [1., 1., -1., -1.],
            [1, -1., -1., 1]
        ], pos=Vector2(0, 2.5), scale=Vector2(2, 2), theta=torch.pi / 4),
        Polygon([
            [1, +1., -1., -1.],
            [1, -1., -1., +1.]
        ], pos=Vector2(1.5, 0), scale=Vector2(0.5, 0.5)),
    ]

    triangle = Polygon([
        [0., 2, -2],
        [2., 0, +0]
    ], pos=Vector2(0, -1))

    for quad in quads:
        axes[2].add_patch(PolygonPatch(quad.world_verts, color='blue'))

    axes[2].add_patch(PolygonPatch(triangle.verts, color='green'))

    quad_tensor = Polygon.stack(quads)

    for q in quad_tensor:
        clipped_q = polygon_clip(q, triangle.verts)
        axes[2].add_patch(PolygonPatch(clipped_q, color='red'))

    axes[2].set_xlim(-5, 5)
    axes[2].set_ylim(-5, 5)

    """
    intersection of 2 line segments
    """

    l1_start = torch.tensor([
        [0., 0.], [0, 0], [0, 0]
    ])
    l1_end = torch.tensor([
        [2., 2.], [1., 2.], [1, 0.5]
    ])

    l2_start = torch.tensor([[0., 3.], [0., 1.]])
    l2_end = torch.tensor([[3., 0.], [1., 0.]])

    plot_line_segments(axes[3], l1_start, l1_end)
    plot_line_segments(axes[3], l2_start, l2_end)

    p, mask = line_seg_intersect(l1_start, l1_end, l2_start, l2_end)

    for p, mask in zip(p.flatten(0, 1), mask.flatten()):
        if mask:
            axes[3].scatter(p[0], p[1])

    """
    raycast
    """

    # ray in parametric form
    ray_origin = torch.tensor([
        [0., 0.], [3, 2], [2, 3]
    ])
    ray_vector = torch.tensor([
        [1., 1.], [-1, -1], [-1, -1]
    ])

    ray_far_end = ray_origin + ray_vector * 4.0

    l2_start = torch.tensor([
        [0., 3.], [0, 2.],
    ])
    l2_end = torch.tensor([
        [3., 0.], [2., 0]])

    plot_line_segments(axes[4], l2_start, l2_end, )
    plot_line_segments(axes[4], ray_origin, ray_far_end, linestyle='dotted')

    ray_origin, ray_end, ray_len = raycast(ray_origin, ray_vector, l2_start, l2_end)

    plot_line_segments(axes[4], ray_origin, ray_end)

    plt.show()
