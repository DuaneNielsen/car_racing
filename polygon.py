import torch
from torch import cos, sin

"""
Line segs are tuples of (N, 2) start, (N, 2) end tensors

If function specifies (N..., 2), you can have any input shape you like,
as long as you make the last dimension 2 (x, y)

If function specifies (... D) then the function will work for any input shape
the last dimension will be consider the number of spacial Dimensions

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
    prev = torch.arange(1, t.size(dim) + 1, device=t.device)
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
    start: (N..., 2) tensor containing N start points
    end: (N..., 2) tensor containing N end points
    clipping_polygons: (P, V, 2) tensor containing P polygons of V vertices with CLOCKWISE winding
    returns:
        (N..., P, 2) tensor of clip enter points
        (N..., P, 2) tensor of clip exit points
        (N..., P) boolean tensor, True if line intersects polygon
    """
    with torch.no_grad():
        # Cyrus Beck algorithm
        # P0 - PEi
        # P0: start point of the line segment
        # P1: end point of the line segment
        # PEi: the vertices of the polygon

        # pack the input tensors
        assert start.shape == end.shape, "start and end must be same shape"
        N = start.shape[:-1]
        start = start.flatten(end_dim=-2)
        end = end.flatten(end_dim=-2)

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

        return p_enter.unflatten(0, N), p_exit.unflatten(0, N), inside.squeeze(-1).unflatten(0, N)


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
        result.v = self.v.clone() + vec2.v
        return result

    def __mul__(self, scalar):
        result = Vector2()
        result.v = self.v.clone()
        result.v = result.v * scalar
        return result

    def __rmul__(self, scalar):
        result = Vector2()
        result.v = self.v.clone()
        result.v = result.v * scalar
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


def scale_matrix(scale):
    scale_x, scale_y = scale[:, 0], scale[:, 1]
    zeros = torch.zeros_like(scale_x)
    ones = torch.ones_like(scale_x)
    return torch.stack([
        torch.stack([scale_x, zeros, zeros], dim=-1),
        torch.stack([zeros, scale_y, zeros], dim=-1),
        torch.stack([zeros, zeros, ones], dim=-1)
    ], dim=1)


def adjoint_matrix(se2):
    """
    Builds an adjoint matrix
    """
    N = se2.shape[0]
    cos_theta = cos(se2[:, 2])
    sin_theta = sin(se2[:, 2])
    x, y = se2[:, 0], se2[:, 1]

    return torch.stack((
        torch.stack([+cos_theta, sin_theta, x], dim=-1),
        torch.stack([-sin_theta, cos_theta, y], dim=-1),
        torch.tensor([0., 0., 1.], device=se2.device).repeat(N).reshape(N, 3)
    ), dim=1)


def apply_transform(transf, verts):
    """
    transf: a (N, 3, 3) homogenous transformation matrix
    verts: (P, V, 2) or (V, 2) list of polygon verts
    returns: (N, V, 2) verts rescaled and transformed
    P and N will need to match, or else specify a single V
    """
    N, _, _ = transf.shape
    if len(verts.shape) == 2:
        verts = verts.unsqueeze(0)
    else:
        assert (N == verts.shape[0]) or (verts.shape[0] == 1) or (N == 1), \
            f"first dimension of verts must match first dimension of transforms (or be equal to 1)"
    P, V, _ = verts.shape
    verts = torch.cat((verts, torch.ones(P, V, 1, device=verts.device)), dim=-1)
    verts = torch.matmul(transf, verts.permute(0, 2, 1)).permute(0, 2, 1)
    return verts[..., 0:2]


def transform_matrix(se2, scale=None):
    """
    se2: (N, 3) vector in SE2, [x, y, theta]
    scale (optional): (N, 2) scaling vector in [x, y]
    returns: (N, 3, 3) transformation matrix
    """
    transform_matrix = adjoint_matrix(se2)

    if scale is not None:
        transform_matrix = torch.matmul(transform_matrix, scale_matrix(scale))

    return transform_matrix


class Camera:
    def __init__(self, se2, scale):
        self._se2 = se2 if len(se2.shape) == 2 else se2.unsqueeze(0)
        self._scale = scale if len(scale.shape) == 2 else scale.unsqueeze(0)

    @property
    def se2(self):
        return self._se2

    @se2.setter
    def se2(self, se2):
        self._se2 = se2 if len(se2.shape) == 2 else se2.unsqueeze(0)

    @property
    def scale(self):
        return self._scale

    @scale.setter
    def scale(self, scale):
        self._scale = scale if len(scale.shape) == 2 else scale.unsqueeze(0)

    def transform(self, verts):
        s_matrix = scale_matrix(self._scale)
        a_matrix = adjoint_matrix(self._se2)
        return apply_transform(s_matrix, apply_transform(a_matrix, verts))

    def to(self, device):
        self._se2 = self._se2.to(device)
        self._scale = self._scale.to(device)
        return self

class Model:
    def __init__(self, verts, N=None):
        """
        Stores a set of model vertices and positions
        verts: polygons are written as 2 lists of co-ordinates [[x1, x2, x3], [y1, y2, y3]]
        you can specify just a single polygon, or a list of polygons
        N: required if you specify a single polygon

        Use CLOCKWISE winding for vertices

        x4, y4 -----------  x1, y1
          -                   -
          -                   -
          -                   -
          -                   -
        x3, y3 ------------ x2, y2

        square = Model([
          [x1, x2, x3, x4]
          [y1, y2, y3, y4]
        ])

        """
        self.verts = torch.tensor(verts)
        if len(self.verts.shape) == 2:
            self.verts = self.verts.T.unsqueeze(0)
            assert N is not None, f"set N to a value if using a single set of verts"
        else:
            self.verts = self.verts.permute(0, 2, 1)
            N = len(verts)
        self.se2 = torch.zeros(N, 3)
        self.scale = torch.ones(N, 2)
        self.parent = None
        self.children = []

    @property
    def N(self):
        """
        returns the number of model instances
        """
        return self.se2.shape[0]

    @property
    def __len__(self):
        return self.N

    @property
    def pos(self):
        """
        returns N, 2 model positions
        """
        return self.se2[:, 0:2]

    @pos.setter
    def pos(self, pos):
        """
        pos: (N, 2)
        """
        self.se2[:, 0:2] = pos

    @property
    def theta(self):
        """
        returns theta angle in radians
        """
        return self.se2[:, 2]

    @theta.setter
    def theta(self, theta):
        """
        theta: (N) in radians
        """
        self.se2[:, 2] = theta

    def parents(self):
        parents = []
        parent = self.parent
        while parent is not None:
            parents += [parent]
            parent = parent.parent
        return parents

    def world_verts(self):
        t_matrix = transform_matrix(self.se2, self.scale)
        for parent in self.parents():
            parent_t_matrix = transform_matrix(parent.se2, parent.scale)
            t_matrix = torch.matmul(parent_t_matrix, t_matrix)
        return apply_transform(t_matrix, self.verts)

    def attach(self, model):
        model.parent = self
        self.children.append(model)

    def to(self, device):
        self.verts = self.verts.to(device)
        self.se2 = self.se2.to(device)
        self.scale = self.scale.to(device)
        return self


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
        self.se2 = torch.zeros(3)

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

    def world_transform(self):
        return translate2D(rotate2D(scale2D(torch.eye(3), self.scale), self._theta), self.pos)

    @property
    def world_verts(self):
        M = translate2D(rotate2D(scale2D(torch.eye(3), self.scale), self._theta), self.pos)
        return torch.matmul(M, self.verts_homo).T[:, 0:2]

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
    t_start: (N..., 2) start points of segments
    t_end: (N..., 2) end points of set of line segments
    u_start: (M..., 2) start points of segment set to intersect
    u_end: (M..., 2) end points of segment set to intersections
    returns: p, mask
        p: (M..., N..., 2)
        mask: (M..., N...)
    """

    # pack leading dimensions
    assert t_end.shape == t_end.shape
    assert u_start.shape == u_end.shape

    N = t_start.shape[:-1]
    M = u_start.shape[:-1]

    t_start, t_end = t_start.flatten(end_dim=-2), t_end.flatten(end_dim=-2)
    u_start, u_end = u_start.flatten(end_dim=-2), u_end.flatten(end_dim=-2)

    t_m, t_b = to_parametric(t_start, t_end)
    u_m, u_b = to_parametric(u_start, u_end)

    t, u, t_m, t_b, u_m, u_b = compute_tu(t_m, t_b, u_m, u_b)
    intersect = t.ge(0.) & t.le(1.) & u.ge(0.) & u.le(1.)
    p = t_m * t + t_b

    # unpack leading dimensions
    p = p.unflatten(1, N)
    p = p.unflatten(0, M)
    intersect = intersect.squeeze(-1)
    intersect = intersect.unflatten(1, N)
    intersect = intersect.unflatten(0, M)

    return p, intersect


def unflatten_index(index, shape):
    """
    takes an index from a flattened tensor and converts the indices
    so they can be applied to the unflattened version of the tensor
    index: (N...) the index to unflatten
    shape: the shape of the unflat tensor
    returns tuple((N)LongTensor, (N)LongTensor), (N)LongTensor))

    >>> x = torch.rand(2, 3)
    >>> x_flat = x.flatten()
    >>> flat_index = torch.arange(2*3)
    >>> x_flat[flat_index]
    tensor([-0.4634, -0.4368,  2.6794,  1.0621, -0.0302,  1.2309])
    >>> x[unflatten_index(flat_index, (2, 3))]
    tensor([-0.4634, -0.4368,  2.6794,  1.0621, -0.0302,  1.2309])
    """

    denominator = 1
    co_ordinates = []
    for i in reversed(shape):
        co_ordinates += [(index.flatten() // denominator) % i]
        denominator *= i
    return tuple(reversed(co_ordinates))


def raycast(ray_origin, ray_vector, l_start, l_end, max_len=None):
    """
    casts a ray against a set of line segments
    ray_origin: (N..., 2) ray origins
    ray_vector: (N..., 2) ray vectors
    l_start: (M..., 2) line segment starts to cast onto
    l_end: (M..., 2) line segment ends to cast onto
    max_len: the length to set the ray if it doesn't hit anything
    returns:
        ray_origin: (N..., 2)
        ray_end: (N..., 2)
        ray_hit: N - True if the ray hit a line segment, False if it didn't
        ray_len: N - contains the length of the tensor
        ray_index: tuple(LongTensors) containing the indices of the hit line_seg
    """

    assert ray_origin.shape == ray_vector.shape, "ray_origin and ray_vector must have the same shape"
    assert l_start.shape == l_end.shape, "l_start and l_end must have the same shape"

    N, M = ray_origin.shape[:-1], l_start.shape[:-1]
    ray_origin, ray_vector = ray_origin.flatten(end_dim=-2), ray_vector.flatten(end_dim=-2)
    l_start, l_end = l_start.flatten(end_dim=-2), l_end.flatten(end_dim=-2)

    t_m, t_b = to_parametric(l_start, l_end)
    ray_vector = ray_vector / torch.linalg.vector_norm(ray_vector, dim=-1, keepdim=True)
    t, r, t_m, t_b, ray_vector, ray_origin = compute_tu(t_m, t_b, ray_vector, ray_origin)

    if max_len is None:
        ray_hit = t.ge(0.) & t.lt(1.0) & r.ge(0.)
        r[~ray_hit] = torch.inf
    else:
        ray_hit = t.ge(0.) & t.lt(1.0) & r.ge(0.) & r.lt(max_len)
        r[~ray_hit] = max_len

    ray_length, ray_index = torch.min(r, dim=1, keepdim=True)
    ray_end = ray_origin + ray_length * ray_vector

    # not sure if this is good
    if len(M) > 0:
        ray_index = unflatten_index(ray_index.squeeze(), shape=M)

    ray_origin, ray_end, ray_len = ray_origin.squeeze(1), ray_end.squeeze(1), ray_length.squeeze()
    ray_origin, ray_end, ray_len = ray_origin.unflatten(0, N), ray_end.unflatten(0, N), ray_len.unflatten(0, N)
    return ray_origin, ray_end, ray_len, ray_hit, ray_index


if __name__ == '__main__':

    from matplotlib import pyplot as plt
    from matplotlib.patches import Polygon as PolygonPatch
    from math import radians

    # plotting
    fig, axes = plt.subplots(2, 4)
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
    axes[0].set_title('polygon clipping with lines')

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
    axes[1].set_title('polygon clipping with lines')

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
    axes[2].set_title('polygon clipping with a polygon')

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
    axes[3].set_title('line group intersections')

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
    axes[4].set_title('raycasting')

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

    ray_origin, ray_end, ray_len, ray_hit, ray_index = raycast(ray_origin, ray_vector, l2_start, l2_end)

    plot_line_segments(axes[4], ray_origin, ray_end)

    """
    se2 transforms
    """
    axes[5].set_title('SE2 style transforms')

    quads = Model([
        [1., 1., -1., -1.],
        [1, -1., -1., 1]
    ], N=4)

    quads.pos = torch.tensor([[0, 0], [3, 3], [-3, -3], [-5, -5]])

    for p in quads.world_verts():
        axes[5].add_patch(PolygonPatch(p, color='red'))

    tri = torch.tensor([
        [0., 1., -1.],
        [1, 0., 0.]
    ]).T

    tri = torch.stack([tri, tri, tri])
    se2 = torch.tensor([[5, 5, radians(34)], [3, -3, radians(60.)], [-5, 5, radians(90.)]])
    scale = torch.ones(3, 2) * 3.

    project = apply_transform(transform_matrix(se2, scale), tri)
    for p in project:
        axes[5].add_patch(PolygonPatch(p, color='red'))
    axes[5].set_xlim(-8, 8)
    axes[5].set_ylim(-8, 8)

    """
    model heirarchy
    """

    axes[6].set_title('model heirarchy')
    axes[6].set_xlim(-5, 5)
    axes[6].set_ylim(-5, 5)

    tanks = Model([
        [1., 1., -1., -1.],
        [1, -1., -1., 1]
    ], N=4)

    turrets = Model([
        [0., 1., -1.],
        [1, 0., 0.]
    ], N=4)

    tanks.attach(turrets)
    tanks.pos = torch.tensor([[3., 3], [3, -3], [-3, -3], [-3, 3]])
    tanks.theta = torch.tensor([radians(-45), radians(0), radians(45.), radians(90.)])
    turrets.theta = torch.tensor([radians(0), radians(90.), radians(180.), radians(270)])

    for tank in tanks.world_verts():
        tank_patch = PolygonPatch(tank, color='blue')
        axes[6].add_patch(tank_patch)

    for turret in turrets.world_verts():
        turrent_patch = PolygonPatch(turret, color='green')
        axes[6].add_patch(turrent_patch)

    plt.show()
