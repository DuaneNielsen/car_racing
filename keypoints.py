import cv2
import numpy as np


def gradient(img, dx, dy, ksize=3):
    deriv_filter = cv2.getDerivKernels(dx=dx, dy=dy, ksize=ksize, normalize=True)
    return cv2.sepFilter2D(img, -1, deriv_filter[0], deriv_filter[1])


def march_map(sdf):
    """
    Constructs a 2D vector field containing the distance to the nearest surface,
    this is done by following the gradient by the distance in the sdf
    :param sdf: signed distance field, H, W
    :return: x, y the distance to the surface in the h and w dirmensions
    """
    sdf_gx = gradient(sdf, dx=1, dy=0)
    sdf_gy = gradient(sdf, dx=0, dy=1)
    y = sdf * - sdf_gy
    x = sdf * - sdf_gx
    return x, y


def extract_kp(src_sdf, sample_index):
    """
    computes a set of corresponding points for two sdfs by grid sampling
    and following the sdf gradient to the nearest surface

    this should work well when sensors are running a rate much faster than
    the robot angular or translational speed

    :param src_sdf:
    :param sample_index: 2, N index array of co-ordinates
    """

    # compute the march maps
    src_map_x, src_map_y = march_map(src_sdf)

    # move each sampled point to the surface using the march map
    src_x_c = sample_index[0] + src_map_x[sample_index[1], sample_index[0]]
    src_y_c = sample_index[1] + src_map_y[sample_index[1], sample_index[0]]

    # stack the results in an (h, w), N
    return np.stack([src_x_c, src_y_c])