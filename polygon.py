import numpy as np
import warnings
from pygame import Vector2


def to_parametric(start, end):
    m = end - start
    b = start
    return m, b


def normal(start, end):
    m, b = to_parametric(start, end)
    return m.norm()


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

    intersection = (x, y)

    return intersection

"""
Given a subject polygon defined by the vertices in clockwise order
subject_polygon = [(x_1,y_1),(x_2,y_2),...,(x_N,y_N)]
and a clipping polygon, which will be used to clip the subject polygon,
defined by the vertices in clockwise order
clipping_polygon = [(x_1,y_1),(x_2,y_2),...,(x_K,y_K)]
and assuming that the subject polygon and clipping polygon overlap,
the Sutherland-Hodgman algorithm works as follows:
for i = 1 to K:

    # this will  store the vertices of the final clipped polygon
    final_polygon = []

    # these two vertices define a line segment (edge) in the clipping
    # polygon. It is assumed that indices wrap around, such that if
    # i = 1, then i - 1 = K.
    c_vertex1 = clipping_polygon[i]
    c_vertex2 = clipping_polygon[i - 1]

    for j = 1 to N:

        # these two vertices define a line segment (edge) in the subject
        # polygon. It is assumed that indices wrap around, such that if
        # j = 1, then j - 1 = N.
        s_vertex1 = subject_polygon[j]
        s_vertex2 = subject_polygon[j - 1]

        # next, we want to check if the points s_vertex1 and s_vertex2 are
        # inside the clipping polygon. Since the points that define the
        # edges of the clipping polygon are listed in clockwise order in
        # clipping_polygon, then we can do this by checking if s_vertex1
        # and s_vertex2 are to the right of the line segment defined by
        # the points (c_vertex1,c_vertex2).
        #
        # if both s_vertex1 and s_vertex2 are inside the clipping polygon,
        # then s_vertex2 is added to the final_polygon list.
        #
        # if s_vertex1 is outside the clipping polygon and s_vertex2 is
        # inside the clipping polygon, then we first add the point of
        # intersection between the edge defined by (s_vertex1,s_vertex2)
        # and the edge defined by (c_vertex1,c_vertex2) to final_polygon,
        # and then we add s_vertex2 to final_polygon.
        #
        # if s_vertex1 is inside the clipping polygon and s_vertex2 is
        # outside the clipping polygon, then we add the point of
        # intersection between the edge defined by (s_vertex1,s_vertex2)
        # and the edge defined by (c_vertex1,c_vertex2) to final_polygon.
        #
        # if both s_vertex1 and s_vertex2 are outside the clipping polygon,
        # then neither are added to final_polygon.
        #
        # note that since we only compute the point of intersection if
        # we know that the edge of the clipping polygon and the edge of
        # the subject polygon intersect, then we can treat them as infinite
        # lines and use the formula given here:
        #
        # https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line
        #
        # to compute the point of intersection.
"""


# POINTS NEED TO BE PRESENTED CLOCKWISE OR ELSE THIS WONT WORK


class PolygonClipper:

    def __init__(self, warn_if_empty=True):
        self.warn_if_empty = warn_if_empty

    def is_inside(self, p1, p2, q):
        R = (p2[0] - p1[0]) * (q[1] - p1[1]) - (p2[1] - p1[1]) * (q[0] - p1[0])
        if R <= 0:
            return True
        else:
            return False

    def clip(self, subject_polygon, clipping_polygon):

        final_polygon = subject_polygon.copy()

        for i in range(len(clipping_polygon)):

            # stores the vertices of the next iteration of the clipping procedure
            next_polygon = final_polygon.copy()

            # stores the vertices of the final clipped polygon
            final_polygon = []

            # these two vertices define a line segment (edge) in the clipping
            # polygon. It is assumed that indices wrap around, such that if
            # i = 1, then i - 1 = K.
            c_edge_start = clipping_polygon[i - 1]
            c_edge_end = clipping_polygon[i]

            for j in range(len(next_polygon)):

                # these two vertices define a line segment (edge) in the subject
                # polygon
                s_edge_start = next_polygon[j - 1]
                s_edge_end = next_polygon[j]

                if self.is_inside(c_edge_start, c_edge_end, s_edge_end):
                    if not self.is_inside(c_edge_start, c_edge_end, s_edge_start):
                        intersection = compute_intersection(s_edge_start, s_edge_end, c_edge_start, c_edge_end)
                        final_polygon.append(intersection)
                    final_polygon.append(tuple(s_edge_end))
                elif self.is_inside(c_edge_start, c_edge_end, s_edge_start):
                    intersection = compute_intersection(s_edge_start, s_edge_end, c_edge_start, c_edge_end)
                    final_polygon.append(intersection)

        return np.asarray(final_polygon)

    def __call__(self, A, B):
        clipped_polygon = self.clip(A, B)
        if len(clipped_polygon) == 0 and self.warn_if_empty:
            warnings.warn("No intersections found. Are you sure your \
                          polygon coordinates are in clockwise order?")

        return clipped_polygon


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
    R = np.array([
        [np.cos(theta), np.sin(theta), 0],
        [-np.sin(theta), np.cos(theta), 0],
        [0., 0., 1.]
    ])
    return np.matmul(R, verts)


def translate2D(verts, vec2):
    T = np.array([
        [1., 0, vec2.x],
        [0., 1., vec2.y],
        [0., 0., 1.]
    ])
    return np.matmul(T, verts)


def scale2D(verts, vec2):
    S = np.array([
        [vec2.x, 0, 0.],
        [0., vec2.y, 0],
        [0., 0., 1.]
    ])
    return np.matmul(S, verts)


def to_pygame_poly(verts):
    return [v for v in zip(verts[0], verts[1])]


def from_pygame_poly(pygame_verts):
    """
    converts pygame format [(x1, y1), (x2, y2)] to homogenous co-ords
    """
    v = np.array(pygame_verts).T
    o = np.ones(len(pygame_verts)).reshape(1, len(pygame_verts))
    return np.concatenate((v, o))


def get_intersect(A, B, C, D):
    '''
    finding intersect point of line AB and CD
    where A is the first point of line AB
    and B is the second point of line AB
    and C is the first point of line CD
    and D is the second point of line CD
    '''

    # a1x + b1y = c1
    a1 = B.y - A.y
    b1 = A.x - B.x
    c1 = a1 * (A.x) + b1 * (A.y)

    # a2x + b2y = c2
    a2 = D.y - C.y
    b2 = C.x - D.x
    c2 = a2 * (C.x) + b2 * (C.y)

    # determinant
    det = a1 * b2 - a2 * b1

    # parallel line
    if det == 0:
        return (float('inf'), float('inf'))

    # intersect point(x,y)
    x = ((b2 * c1) - (b1 * c2)) / det
    y = ((a1 * c2) - (a2 * c1)) / det
    return (x, y)


class Polygon:
    def __init__(self, verts):
        """
        Use CLOCKWISE winding for vertices
        vertices is numpy array in homogenous coordinates

        x4, y4 -----------  x1, y1
          -                   -
          -                   -
          -                   -
          -                   -
        x3, y3 ------------ x2, y2

        np.array([
          [x1, x2, x3, x4]
          [y1, y2, y3, y4]
          [1., 1., 1., 1.]
        ])

        """
        self.verts = verts
        self.pos = Vector2(0., 0.)
        self.theta = 0.
        self.scale = Vector2(1., 1.)

    @property
    def world_verts(self):
        return translate2D(rotate2D(scale2D(self.verts, self.scale), self.theta), self.pos)

    @property
    def pygame_world_verts(self):
        return to_pygame_poly(self.world_verts)
