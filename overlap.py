import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import cv2
import math


class Frame:
    def __init__(self, x=0.0, y=0.0, theta=0.0):
        self.height = 20
        self.width = 30
        self.i = np.zeros((self.height, self.width))
        self.i[0, :] = 1.0
        self.i[self.height - 1, :] = 1.0
        self.i[:, 0] = 1.0
        self.i[:, self.width - 1] = 1.0
        self._R = None
        self.theta = theta
        self._t = np.array([x, y]).reshape(2, 1)
        self.vertices = np.array([
            [0, 0],
            [self.width, 0],
            [self.width, self.height],
            [0, self.height]
        ])

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
    def R(self):
        return self._R

    @R.setter
    def R(self, R):
        self._R = R

    @property
    def theta(self):
        return math.acos(self._R[0, 0])

    @theta.setter
    def theta(self, theta):
        self._R = np.array([
            [math.cos(theta), math.sin(theta)],
            [-math.sin(theta), math.cos(theta)]
        ])

    @property
    def M(self):
        """
        transforms to world space
        :return: homogenous matrix in SE2
        """
        M = np.concatenate((self.R, self._t), axis=1)
        return np.concatenate([M, np.array([0., 0., 1]).reshape(1, 3)], axis=0)

    @property
    def inv_M(self):
        """
        transforms from world space to object space
        :return:  inverse homogenous matrix in SE2
        """
        M = np.concatenate((self.R.T, np.matmul(-self.R.T, self._t)), axis=1)
        return np.concatenate([M, np.array([0., 0., 1]).reshape(1, 3)], axis=0)

    def world_image(self, world_shape):
        return cv2.warpAffine(self.i, self.M, world_shape)


def grid_sample(x, y, grid_spacing, pad=0):
    """

    :param x: length of grid in x (width)
    :param y: length of grid in y (height)
    :param grid_spacing: spacing between samples
    :param pad: padding at the edges
    :return: N, 2 array of x, y integer co-ordinates
    """
    x, y = x - 1 - pad, y - 1 - pad
    x_i = np.floor(np.linspace(pad, x, x // grid_spacing)).astype(int)
    y_i = np.floor(np.linspace(pad, y, y // grid_spacing)).astype(int)
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
    A = line_eqns[0].reshape(V, 1) * P[0].reshape(1, N)
    B = line_eqns[1].reshape(V, 1) * P[1].reshape(1, N)
    C = line_eqns[2].reshape(V, 1) * np.ones(N).reshape(1, N)

    # combine to get the side
    side = A + B + C

    # if on the left, the point is good
    inside = side >= 0.0

    # if all points are to the left of the polygon lines, the point is inside
    return np.all(inside, axis=0)


if __name__ == '__main__':

    fig = plt.figure()
    r1, r2, world = fig.subplots(1, 3)
    fig.show()

    f1 = Frame()
    assert np.allclose(np.matmul(f1.M, f1.inv_M), np.eye(3))

    f2 = Frame(x=5, y=8, theta=math.radians(30))
    assert np.allclose(np.matmul(f2.inv_M, f2.M), np.eye(3))

    grid = grid_sample(f1.width, f1.height, grid_spacing=4, pad=2)
    M = np.matmul(f1.inv_M, f2.M)
    grid_f2 = transform_points(M, grid.T).T

    r1.imshow(f1.i, origin='lower')
    r2.imshow(f2.i, origin='lower')
    for i in range(len(grid)):
        r1.scatter(grid[i, 0], grid[i, 1])
        r2.scatter(grid_f2[i, 0], grid_f2[i, 1])
    r1.set_aspect('equal')
    r2.set_aspect('equal')

    wf1 = transform_points(f1.M, f1.vertices.T)
    wf2 = transform_points(f2.M, f2.vertices.T)
    inside = inside(grid.T, wf2)
    wf1 = Polygon(wf1.T, color=[1, 0, 0], fill=False)
    wf2 = Polygon(wf2.T, color=[0, 1, 0], fill=False)
    world.add_patch(wf1)
    world.add_patch(wf2)
    world.scatter(grid[inside, 0], grid[inside, 1], color='blue')
    world.scatter(grid[~inside, 0], grid[~inside, 1], color='red')
    world.set_aspect('equal')
    world.autoscale_view()

    plt.show()