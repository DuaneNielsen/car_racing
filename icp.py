import numpy as np
from numpy.linalg import svd


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