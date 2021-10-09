import numpy as np


def R(theta):
    return np.array([
        [np.cos(theta), np.sin(theta)],
        [-np.sin(theta), np.cos(theta)]
    ])


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

    @property
    def inv_M(self):
        """
        transforms from world space to object space
        :return:  inverse homogenous matrix in SE2
        """
        M = np.concatenate((self._R.T, np.matmul(-self._R.T, self._t)), axis=1)
        return np.concatenate([M, np.array([0., 0., 1]).reshape(1, 3)], axis=0)


class Scan(Frame):
    def __init__(self, height, width, x=0.0, y=0.0, theta=0.0):
        super().__init__()
        self.x = x
        self.y = y
        self.theta = theta
        self.h = height
        self.w = width
        self.i = np.zeros((self.h, self.w))
        self.i[0, :] = 1.0
        self.i[self.h - 1, :] = 1.0
        self.i[:, 0] = 1.0
        self.i[:, self.w - 1] = 1.0
        self.vertices = np.array([
            [0, 0],
            [self.w, 0],
            [self.w, self.h],
            [0, self.h]
        ])


class Keypoints(Frame):
    def __init__(self, P, center, R, scan):
        """

        :param P: 2, N kepoints in scan co-ordinate space
        :param center: the centroid of the keypoints in scan co-ordinate space
        :param R: rotation about the center
        :param scan: the scan the keypoints are from
        """
        super().__init__()
        self.P = P
        self.t = center
        self.R = R
        self.scan = scan
        self.height = 20
        self.width = 30


def grid_sample(h, w, grid_spacing, pad=0):
    """

    :param w: width of grid
    :param h: height of grid
    :param grid_spacing: spacing between samples
    :param pad: padding at the edges
    :return: N, 2 array of x, y integer co-ordinates
    """
    w, h = w - 1 - pad, h - 1 - pad
    x_i = np.floor(np.linspace(pad, w, w // grid_spacing)).astype(int)
    y_i = np.floor(np.linspace(pad, h, h // grid_spacing)).astype(int)
    x_i, y_i = np.meshgrid(x_i, y_i)
    return np.stack([x_i.flatten(), y_i.flatten()], axis=1)


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
    :param convex_poly: 2, V list of polygon vertices wound counter-clockwise
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
