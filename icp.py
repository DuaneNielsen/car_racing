import numpy as np
from numpy.linalg import svd
import numpy.random


"""
Iterative closes point algorithms 
"""


def rms(source, target):
    return np.sum((source - target) ** 2) / source.shape[1]


def icp(source, target):
    """

    1 iteration of icp algorithm

    :param source: set of source points, D, N
    :param target: corresponding set of target points, D, N
    :return: rotation matrix, translation from Source -> Target

    note: the translation output by this function already accounts for the rotation
    """
    _, N = source.shape

    source_center = source.mean(axis=1, keepdims=True)
    target_center = target.mean(axis=1, keepdims=True)
    corr = np.matmul(target - target_center, (source - source_center).T)
    u, d, v = svd(corr)
    R = np.matmul(u, v.T)
    t = target_center - np.matmul(R, source_center)
    return R, t


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
        samples += [rng.choice(N, n, replace=False)]
    samples = np.concatenate(samples)
    source = source[:, samples].reshape(2, n, k)
    source = np.moveaxis(source, 2, 0)
    target = target[:, samples].reshape(2, n, k)
    target = np.moveaxis(target, 2, 0)
    return source, target


def ransac_icp(source, target, k, n):
    source, target = ransac_sample(source, target, k, n)

    best_error = np.inf
    best_R = None
    best_t = None
    best_k = k
    for k in range(k):
        R, t = icp(source[k], target[k])
        error = rms(np.matmul(R, source[k] + t), target[k])
        if error < best_error:
            best_error = error
            best_R = R
            best_t = t
            best_k = k
    return best_R, best_t, source[best_k], target[best_k], best_error
