import cv2
import numpy as np


"""
Extracts key-points from Signed Distance Field
"""


def gradient(img, dx, dy, ksize=5):
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


def extract_kp(src_sdf, sample_index, iterations=2):
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

    for _ in range(iterations):
        # move each sampled point to the surface using the march map
        src_x_c = sample_index[0] + src_map_x[sample_index[1], sample_index[0]]
        src_y_c = sample_index[1] + src_map_y[sample_index[1], sample_index[0]]
        sample_index = np.round(np.stack([src_x_c, src_y_c])).astype(int)

    return sample_index


def extract_cv2_kp(extractor, query, train, match_metric=cv2.NORM_HAMMING):
    query_keypoints, query_descriptors = extractor.detectAndCompute(query, None)
    train_keypoints, train_descriptors = extractor.detectAndCompute(train, None)
    bf = cv2.BFMatcher(match_metric, crossCheck=True)
    matches = bf.match(query_descriptors, train_descriptors)
    matches = sorted(matches, key=lambda x: x.distance)
    kp_0, kp_1 = [], []
    for m in matches:
        kp_0.append(np.array(query_keypoints[m.queryIdx].pt))
        kp_1.append(np.array(train_keypoints[m.trainIdx].pt))
    kp_0 = np.stack(kp_0, axis=1)
    kp_1 = np.stack(kp_1, axis=1)
    return kp_0, kp_1



