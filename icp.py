import numpy as np
from numpy.linalg import svd
import numpy.random
import geometry as geo
from copy import copy

"""
Iterative closes point algorithms 
"""


def rms(source, target, reduce=True):
    distance = np.sqrt(((source - target) ** 2).sum(axis=0))
    if reduce:
        return distance.mean()
    else:
        return distance


def rms_frames(source, target, reduce=True):
    source_kp_w = geo.transform_points(source.M, source.kp)
    target_kp_w = geo.transform_points(target.M, target.kp)
    return rms(source_kp_w, target_kp_w, reduce)


def icp(source, target):
    """

    1 iteration of icp algorithm

    :param source: set of source points, D, N in the source frame
    :param target: corresponding set of target points, D, N in the target frame
    :return: rotation matrix, translation from Source -> Target

    to apply, translate first, then rotate

    """
    _, N = source.shape

    source_center = source.mean(axis=1, keepdims=True)
    target_center = target.mean(axis=1, keepdims=True)
    corr = np.matmul(source - source_center, (target - target_center).T)
    u, d, v = svd(corr)
    R = np.matmul(u, v.T)
    t = source_center - np.matmul(R, target_center)
    return R, t


def icp_homo(source, target):
    """

    1 iteration of icp algorithm

    :param source_frame: source keypoints
    :param target_frame: target keypoints
    :return: homogenous transform that aligns source kp -> target kp

    note: to apply to frame, put keypoints in world space, then the resulting
    homo transform will also be in world space
    """

    _, N = source.shape

    source_center = source.mean(axis=1, keepdims=True)
    target_center = target.mean(axis=1, keepdims=True)
    corr = np.matmul((target - target_center), (source - source_center).T)
    u, d, v = svd(corr)
    R = np.matmul(u, v.T)

    # rotate the source around the centroid to align with the target frame
    M = geo.R_around_point_Rt(R, source_center)

    # translate source frame to target frame
    M[0:2, 2:] += target_center - source_center

    return M


def ransac_sample(source, target, k, n, seed=None):
    """
    samples corresponding pairs at random

    :param source: 2, N of source points
    :param target: 2, N of corresponding target points
    :param k: the number of samples to generate
    :param n: the number pf points in each sample
    :return: source, target -> k, 2, n
    """
    _, N = source.shape
    rng = numpy.random.default_rng(seed)
    samples = []
    for _ in range(k):
        try:
            samples += [rng.choice(N, n, replace=False)]
        except ValueError:
            print(f'tried to draw {n} samples from {N} points and failed')
    samples = np.concatenate(samples)
    source = source[:, samples].reshape(2, n, k)
    source = np.moveaxis(source, 2, 0)
    target = target[:, samples].reshape(2, n, k)
    target = np.moveaxis(target, 2, 0)
    return source, target


def ransac_icp(source, target, k, n, threshold, d, seed=None):
    source_sample, target_sample = ransac_sample(source, target, k, n, seed)

    best_error = np.inf
    best_M = icp_homo(source, target)
    best_inliers = np.ones(source.shape[1], dtype=np.bool8)

    for k in range(source_sample.shape[0]):
        M = icp_homo(source_sample[k], target_sample[k])
        distance = rms(geo.transform_points(M, source), target, reduce=False)
        inlier_indx = distance < threshold
        N_inliers = np.count_nonzero(inlier_indx)

        if N_inliers > d:
            inlier_source, inlier_target = source[:, inlier_indx], target[:, inlier_indx]
            M = icp_homo(inlier_source, inlier_target)
            error = rms(geo.transform_points(M, inlier_source), inlier_target)

            if error < best_error:
                best_error = error
                best_M = M
                best_inliers = inlier_indx

    return best_M, best_inliers, best_error


def filter_kp(t0_kp_world, t0_rect_world, t1_kp_world, t1_rect_world):
    t0_kp_w, t1_kp_w, intersection = geo.clip_intersection(t0_kp_world, t0_rect_world, t1_kp_world, t1_rect_world)

    # filter non unique key-points
    unique_t0 = geo.naive_unique(t0_kp_world)
    unique_t1 = geo.naive_unique(t1_kp_world)

    filter = intersection & unique_t0 & unique_t1

    return t0_kp_world[:, filter], t1_kp_world[:, filter], filter


def update_frame(t0, t1, k, n, threshold, d):

    # project key-points to world space and clip key-points outside the scan overlap
    t0_kp_w, t1_kp_w, filter = filter_kp(t0.kp_w, t0.vertices_w, t1.kp_w, t1.vertices_w)

    rms_before = rms(t1_kp_w, t0_kp_w)

    # compute alignment and update the t1 frame
    M, inliers, error = ransac_icp(source=t1_kp_w, target=t0_kp_w, k=k, n=n, threshold=threshold, d=d)
    t1.M = np.matmul(M, t1.M)

    return t1, M, inliers, {'rms': error, 'rms_before': rms_before}