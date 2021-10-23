import numpy as np
from numpy.linalg import svd
import numpy.random
import geometry as geo


"""
Iterative closes point algorithms 
"""


def rms(source, target, reduce=True):
    distance = np.sqrt(((source - target) ** 2).sum(axis=0))
    if reduce:
        return distance.mean()
    else:
        return distance


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


def icp_homo(source_frame, target_frame):
    """

    1 iteration of icp algorithm

    :param source_frame: source frame to align
    :param target_frame: target frame to align with
    :return: source_frame aligned to target frame
    """

    source = geo.transform_points(source_frame.M, source_frame.kp)
    target = geo.transform_points(target_frame.M, target_frame.kp)

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

    source_frame.M = np.matmul(M, source_frame.M)

    return source_frame


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
    best_R = None
    best_t = None
    best_inliers = None

    for k in range(source_sample.shape[0]):
        R, t = icp(source_sample[k], target_sample[k])
        distance = rms(np.matmul(R, source + t), target, reduce=False)
        inlier_indx = distance < threshold
        N_inliers = np.count_nonzero(inlier_indx)

        if N_inliers > d:
            inlier_source, inlier_target = source[:, inlier_indx], target[:, inlier_indx]
            R, t = icp(inlier_source, inlier_target)
            error = rms(np.matmul(R, inlier_source + t), inlier_target)

            if error < best_error:
                best_error = error
                best_R = R
                best_t = t
                best_inliers = inlier_indx

    return best_R, best_t, best_inliers, best_error


def update_frame(t0, t1, t0_kp, t1_kp, k, n, threshold, d):

    # project key-points to world space and clip key-points outside the scan overlap
    t0_kp_w, t1_kp_w = geo.project_and_clip_kp(t0_kp, t1_kp, t0, t1)
    rms_before = rms(t1_kp_w, t0_kp_w)

    # filter non unique key-points
    unique = geo.naive_unique(t0_kp_w)
    t0_kp_w, t1_kp_w = t0_kp_w[:, unique], t1_kp_w[:, unique]
    unique = geo.naive_unique(t1_kp_w)
    t0_kp_w, t1_kp_w = t0_kp_w[:, unique], t1_kp_w[:, unique]

    t0_kp_t0, t1_kp_t1 = geo.transform_points(t0.inv_M, t0_kp_w), geo.transform_points(t1.inv_M, t1_kp_w)

    # compute alignment and update the t1 frame
    R, t, inliers, error = ransac_icp(t0_kp_t0, t1_kp_t1, k=k, n=n, threshold=threshold, d=d)

    t1.t = t1.t + t
    # t1.x += t[1]
    # t1.y += t[0]
    t1.R = np.matmul(R, t1.R)

    return t1, R, t, {'rms': error, 'rms_before': rms_before, 't0_kp': t0_kp_t0, 't1_kp': t1_kp_t1, 'inliers': inliers}