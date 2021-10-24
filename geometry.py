import numpy as np
from numpy.linalg import norm

"""
2D geometry library for managing key-points
"""


def R(theta):
    return np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])


def theta(R):
    x = np.array([1.0, 0.]).reshape(2, 1)
    x = np.matmul(R, x)
    x = x / norm(x)
    return np.arccos(x[0, 0])


def R_around_point(theta, x=0.0, y=0.0):
    """
    returns a homogenous rotation matrix around point x,y
    :param theta: angle
    :param x: x co-ordinate of point to rotate around
    :param y: y co-ordinate of point to rotate around
    :return: homogenous transformation matrix
    """
    return np.array([
        [np.cos(theta), -np.sin(theta), x - np.cos(theta) * x + np.sin(theta) * y],
        [np.sin(theta), np.cos(theta), y - np.sin(theta) * x + np.cos(theta) * y],
        [0., 0., 1.]
    ])


def R_around_point_from_R(R, x, y):
    """
    returns a homogenous rotation matrix around point x,y

    :param R: rotation matrix
    :param x: x co-ordinate of point to rotate around
    :param y: y co-ordinate of point to rotate around
    :return: homogenous transformation matrix
    """
    tx = x - R[0, 0] * x - R[0, 1] * y
    ty = y - R[1, 0] * x - R[1, 1] * y
    return np.array([
        [R[0, 0], R[0, 1], tx],
        [R[1, 0], R[1, 1], ty],
        [0., 0., 1]
    ])


def R_around_point_Rt(R, t):
    """
    returns a homogenous rotation matrix around point x,y

    :param R: rotation matrix
    :param t: a point to rotate around
    :return: homogenous transformation matrix
    """
    tx = t[0] - R[0, 0] * t[0] - R[0, 1] * t[1]
    ty = t[1] - R[1, 0] * t[0] - R[1, 1] * t[1]
    return np.array([
        [R[0, 0], R[0, 1], tx[0]],
        [R[1, 0], R[1, 1], ty[0]],
        [0., 0., 1]
    ])


class Frame:
    def __init__(self):
        self._R = np.eye(2)
        self._t = np.zeros(2).reshape(2, 1)

    @property
    def x(self):
        return self._t[0]

    @x.setter
    def x(self, x):
        self._t[0] = x

    @property
    def y(self):
        return self._t[1]

    @y.setter
    def y(self, y):
        self._t[1] = y

    @property
    def t(self):
        return self._t

    @t.setter
    def t(self, t):
        self._t = t

    @property
    def R(self):
        return self._R

    @R.setter
    def R(self, R):
        self._R = R

    @property
    def theta(self):
        return np.arccos(self._R[0, 0])

    @theta.setter
    def theta(self, theta):
        self._R = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]
        ])

    @property
    def M(self):
        """
        transforms to world space
        :return: homogenous matrix in SE2
        """
        M = np.concatenate((self._R, self._t), axis=1)
        return np.concatenate([M, np.array([0., 0., 1]).reshape(1, 3)], axis=0)

    @M.setter
    def M(self, M):
        self._R = M[0:2, 0:2]
        self._t = M[0:2, 2:]

    @property
    def inv_M(self):
        """
        transforms from world space to object space
        :return:  inverse homogenous matrix in SE2
        """
        M = np.concatenate((self._R.T, np.matmul(-self._R.T, self._t)), axis=1)
        return np.concatenate([M, np.array([0., 0., 1]).reshape(1, 3)], axis=0)


class Scan(Frame):
    def __init__(self, height, width, image=None, x=0.0, y=0.0, theta=0.0):
        super().__init__()
        self.x = x
        self.y = y
        self.theta = theta
        self.h = height
        self.w = width
        self.image = image
        self.vertices = np.array([
            [0, 0],
            [self.w-1, 0],
            [self.w-1, self.h-1],
            [0, self.h-1]
        ]).T
        self.kp = None

    @property
    def kp_w(self):
        return transform_points(self.M, self.kp)

    @property
    def centroid(self):
        return self.kp.mean(axis=1, keepdims=True)

    @property
    def centroid_w(self):
        return transform_points(self.M, self.centroid)

    @property
    def vertices_w(self):
        return transform_points(self.M, self.vertices)


def grid_sample(h, w, grid_spacing, pad=0):
    """

    :param w: width of grid
    :param h: height of grid
    :param grid_spacing: spacing between samples
    :param pad: padding at the edges
    :return: 2, N array of x, y integer co-ordinates
    """
    w, h = w - 1 - pad, h - 1 - pad
    x_i = np.floor(np.linspace(pad, w, w // grid_spacing)).astype(int)
    y_i = np.floor(np.linspace(pad, h, h // grid_spacing)).astype(int)
    x_i, y_i = np.meshgrid(x_i, y_i)
    return np.stack([x_i.flatten(), y_i.flatten()])


def transform_points(M, P):
    """
    :param M: 3 x 3 homogenous transformation matrix in SE2
    :param P: 2 x N matrix of x, y co-ordinates
    :return: 2 x N matrix of x, y co-ordinates
    """
    P = np.concatenate([P, np.ones((1, P.shape[1]))], axis=0)
    P = np.matmul(M, P)
    return P[0:2]


def line_eqn_coeff(polygon):
    """
    equation of a line is A x + B y + C = 0

    given 2 points on a line a = (x1, y1), b = (x2, y2)

    A = - (y2 - y1)
    B = x2 - x1
    C = - ( A * x1 + B * y1 )

    :param polygon: 2, N matrix of polygon vertices
    :return: 3, N matrix of line segment co-efficients A, B, C
    """
    a = polygon
    b = np.roll(polygon, -1, axis=1)
    A = - (b[1] - a[1])
    B = b[0] - a[0]
    C = -(A * a[0] + B * a[1])
    return np.stack([A, B, C])


def inside(P, convex_poly):
    """
    Check if a list of points are inside a convex polygon
    :param P: 2, N list of points
    :param convex_poly: 2, V list of convex polygon vertices wound counter-clockwise
    :return: 1, N boolean, True if inside poly
    """

    """
    equation of a line is Ax + By + C = 0

    so if we put P (x, y) into the equation, we get a single number out that indicates

    if side == 0 point is on line
    if side > 0 point is left
    if side < 0 point is right of line
    """
    _, N = P.shape
    _, V = convex_poly.shape

    # get the line equation co-efficients A, B and C
    line_eqns = line_eqn_coeff(convex_poly)

    # compute A B and C for every point vs every vertex
    x = P[0].reshape(1, N)
    y = P[1].reshape(1, N)
    A = line_eqns[0].reshape(V, 1)
    B = line_eqns[1].reshape(V, 1)
    C = line_eqns[2].reshape(V, 1) * np.ones(N).reshape(1, N)

    # combine to get the side
    side = A * x + B * y + C

    # true if on the left or on the line
    inside = side >= 0.0

    # if all points are to the left of the polygon lines, the point is inside
    return np.all(inside, axis=0)


def project_and_clip_sample_indices(sample_indices, from_scan, to_scan):
    """
    Takes a 2, N set of indices from a scan (from_scan)
    projects them into the to_scans frame
    clips points that do not lie in the intersection of the frames
    returns a 2, N index tensors for each scan

    :param sample_indices: 2, N set of indices in from_frame
    :param from_scan: the from_scan
    :param to_scan: the to_scan
    :return: sample_indices_in_from_scan, sample_indices_in_to_scan
    """

    # compute projections between the frames
    M = np.matmul(to_scan.inv_M, from_scan.M)

    # project the t0 sample indices into t1s frame
    sample_index_in_to_frame = np.round(transform_points(M, sample_indices)).astype(int)

    # filter sample points that are outside the t1 frame
    grid_t1_s_inside = inside(sample_index_in_to_frame, to_scan.vertices)
    sample_index_in_to_frame = sample_index_in_to_frame[:, grid_t1_s_inside]
    sample_indices = sample_indices[:, grid_t1_s_inside]
    return sample_indices, sample_index_in_to_frame


def clip_intersection(t0_kp_world, t0_rect_world, t1_kp_world, t1_rect_world):
    """
    project corresponding key-points from each frame into world space
    and remove key-points that are outside the overlap of the scan areas

    :param t0_kp_world: 2, N list of key-points in world frame
    :param t0_rect_world: 2, N list of vertices of a bounding convex poly in world frame
    :param t1_kp_world: 2, N list of key-points in world_frame
    :param t1_rect_world: 2, N list of vertices of a bounding convex poly in world frame
    :return: t0_kp_world, t1_kp_world clipped kp in world frame
    """

    # verify kp are inside the intersection
    t0_kp_inside_t0 = inside(t0_kp_world, t0_rect_world)
    t1_kp_inside_t0 = inside(t1_kp_world, t0_rect_world)
    t0_kp_inside_t1 = inside(t0_kp_world, t1_rect_world)
    t1_kp_inside_t1 = inside(t1_kp_world, t1_rect_world)

    intersection = t0_kp_inside_t0 & t1_kp_inside_t0 & t0_kp_inside_t1 & t1_kp_inside_t1

    # filter the kp outside the intersection
    t0_kp_world = t0_kp_world[:, intersection]
    t1_kp_world = t1_kp_world[:, intersection]

    return t0_kp_world, t1_kp_world, intersection


def naive_unique(kp):
    """

    :param kp: 2, N array of keypoints
    :return: index array of boolean, True if point is unique
        non - unique points in sequence are discarded in order of lowest index first..
        ie: input sequence [1, 1, 1, 0] then we would return filter [False, False True, True]
    """

    _, N = kp.shape
    unique = np.ones(N, dtype=np.bool8)
    for i in range(N-1):
        d = np.abs(kp[:, i+1:] - kp[:, i:i+1])
        d = d.sum(axis=0)
        unique[i] = np.all(d != 0)
    return unique