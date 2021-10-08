import numpy as np
from numpy.linalg import svd
import cv2
from matplotlib import pyplot as plt
import math


def grid_sample(h, w, grid_spacing, pad=0):
    h, w = h - 1 - pad, w - 1 - pad
    h_i = np.floor(np.linspace(pad, h, h // grid_spacing)).astype(int)
    w_i = np.floor(np.linspace(pad, w, w // grid_spacing)).astype(int)
    h_m, w_m = np.meshgrid(h_i, w_i)
    return h_m.flatten(), w_m.flatten()


def gradient(img, dx, dy, ksize=3):
    deriv_filter = cv2.getDerivKernels(dx=dx, dy=dy, ksize=ksize, normalize=True)
    return cv2.sepFilter2D(img, -1, deriv_filter[0], deriv_filter[1])


def march_map(sdf):
    """
    Constructs a 2D vector field containing the distance to the nearest surface,
    this is done by following the gradient by the distance in the sdf
    :param sdf: signed distance field, H, W
    :return: h, w the distance to the surface in the h and w dirmensions
    """
    sdf_gw = gradient(sdf, dx=1, dy=0)
    sdf_gh = gradient(sdf, dx=0, dy=1)
    h = sdf * - sdf_gh
    w = sdf * - sdf_gw
    return h, w


def extract_kp(src_sdf, h_i, w_i):
    """
    computes a set of corresponding points for two sdfs by grid sampling
    and following the sdf gradient to the nearest surface

    this should work well when sensors are running a rate much faster than
    the robot angular or translational speed

    :param src_sdf:
    :param h_i: index array of h co-ordinates
    :return: w_i: index array of w co-ordinates
    """

    # compute the march maps
    src_map_h, src_map_w = march_map(src_sdf)

    # move each sampled point to the surface using the march map
    src_h_c = h_i + src_map_h[h_i, w_i]
    src_w_c = w_i + src_map_w[h_i, w_i]

    # stack the results in an (h, w), N
    return np.stack([src_h_c, src_w_c])


def rms(source, target):
    return np.sum((source - target) ** 2) / source.shape[1]


def icp(source, target):
    """

    :param source: set of source points, D, N
    :param target: corresponding set of target points, D, N
    :return: corrected source points, rotation matrix, translation
    """
    source_center = source.mean(axis=1, keepdims=True)
    target_center = target.mean(axis=1, keepdims=True)
    t = - target_center + source_center
    source = target + t
    corr = np.matmul(target, source.T)
    u, d, v = svd(corr)
    R = np.matmul(u, v.T)
    source = np.matmul(R, source)
    return source, R, t


fig = plt.figure(figsize=(18, 10))
axes = fig.subplots(1, 2)
pre_warped_plot, after_warped_plot, = axes
fig.show()


episode = np.load('episode.npy')

# list of sample indices
h_i, w_i = grid_sample(*episode.shape[1:3], grid_spacing=16, pad=12)

t = 30

t0 = episode[t]
t1 = episode[t + 1]

t0_kp = extract_kp(t0, h_i, w_i)
t1_kp = extract_kp(t1, h_i, w_i)
print(rms(t1_kp, t0_kp))

t1_kp_warped = t1_kp.copy()


t1_kp_warped, R, translate = icp(t1_kp_warped, t0_kp)
translate = np.concatenate((np.eye(2), translate), axis=1)
t1_warped = cv2.warpAffine(t1, translate, (t1.shape[1], t1.shape[0]))
rotate = cv2.getRotationMatrix2D(center=(translate[0, 2], translate[1, 2]), angle=math.acos(R[0, 0]), scale=1.0)
t1_warped = cv2.warpAffine(t1_warped, rotate, (t1.shape[1], t1.shape[0]))
print(rms(t1_kp_warped, t0_kp))
t1_kp_warped = extract_kp(t1_warped, h_i, w_i)
print(rms(t1_kp_warped, t0_kp))

t1_kp_warped, R, translate = icp(t1_kp_warped, t0_kp)
translate = np.concatenate((np.eye(2), translate), axis=1)
t1_warped = cv2.warpAffine(t1_warped, translate, (t1.shape[1], t1.shape[0]))
rotate = cv2.getRotationMatrix2D(center=(translate[0, 2], translate[1, 2]), angle=math.acos(R[0, 0]), scale=1.0)
t1_warped = cv2.warpAffine(t1_warped, rotate, (t1.shape[1], t1.shape[0]))
print(rms(t1_kp_warped, t0_kp))
t1_kp_warped = extract_kp(t1_warped, h_i, w_i)
print(rms(t1_kp_warped, t0_kp))

t1_kp_warped, R, translate = icp(t1_kp_warped, t0_kp)
translate = np.concatenate((np.eye(2), translate), axis=1)
t1_warped = cv2.warpAffine(t1_warped, translate, (t1.shape[1], t1.shape[0]))
rotate = cv2.getRotationMatrix2D(center=(translate[0, 2], translate[1, 2]), angle=math.acos(R[0, 0]), scale=1.0)
t1_warped = cv2.warpAffine(t1_warped, rotate, (t1.shape[1], t1.shape[0]))
print(rms(t1_kp_warped, t0_kp))
t1_kp_warped = extract_kp(t1_warped, h_i, w_i)
print(rms(t1_kp_warped, t0_kp))


t1_kp_warped, R, translate = icp(t1_kp_warped, t0_kp)
translate = np.concatenate((np.eye(2), translate), axis=1)
t1_warped = cv2.warpAffine(t1_warped, translate, (t1.shape[1], t1.shape[0]))
rotate = cv2.getRotationMatrix2D(center=(translate[0, 2], translate[1, 2]), angle=math.acos(R[0, 0]), scale=1.0)
t1_warped = cv2.warpAffine(t1_warped, rotate, (t1.shape[1], t1.shape[0]))
print(rms(t1_kp_warped, t0_kp))
t1_kp_warped = extract_kp(t1_warped, h_i, w_i)
print(rms(t1_kp_warped, t0_kp))


t1_kp_warped, R, translate = icp(t1_kp_warped, t0_kp)
translate = np.concatenate((np.eye(2), translate), axis=1)
t1_warped = cv2.warpAffine(t1_warped, translate, (t1.shape[1], t1.shape[0]))
rotate = cv2.getRotationMatrix2D(center=(translate[0, 2], translate[1, 2]), angle=math.acos(R[0, 0]), scale=1.0)
t1_warped = cv2.warpAffine(t1_warped, rotate, (t1.shape[1], t1.shape[0]))
print(rms(t1_kp_warped, t0_kp))
t1_kp_warped = extract_kp(t1_warped, h_i, w_i)
print(rms(t1_kp_warped, t0_kp))

# t1_kp_warped = extract_kp(t1_warped, h_i, w_i)
# prev_rms_error, d_rms_error = rms(t1_kp_warped, t0_kp), 10.0
# print(f'warped svt error {rms(t1_kp_warped, t0_kp)}')
#
# # re-align by icp
# while d_rms_error > 0.01:
#     t1_kp_warped, R, translate = icp(t1_kp_warped, t0_kp)
#     rms_error = rms(t1_kp_warped, t0_kp)
#     d_rms_error = abs(prev_rms_error - rms_error)
#     print(rms_error)
#     prev_rms_error = rms_error
#
# print(rms(t1_kp_warped, t0_kp))


while True:

    # show t0
    [ax.clear() for ax in axes]
    pre_warped_plot.imshow(t0)
    after_warped_plot.imshow(t0)

    # plot the corresponding points
    for i in range(t0_kp.shape[1]):
        pre_warped_plot.scatter(t0_kp[1, i], t0_kp[0, i])
        after_warped_plot.scatter(t0_kp[1, i], t0_kp[0, i])

    fig.canvas.draw()
    plt.pause(1.0)

    # show warp
    [ax.clear() for ax in axes]
    pre_warped_plot.imshow(t1)
    after_warped_plot.imshow(t1_warped)

    # plot the corresponding points
    for i in range(t0_kp.shape[1]):
        pre_warped_plot.scatter(t1_kp[1, i], t1_kp[0, i])
        after_warped_plot.scatter(t1_kp_warped[1, i], t1_kp_warped[0, i])

    fig.canvas.draw()
    plt.pause(1.0)
