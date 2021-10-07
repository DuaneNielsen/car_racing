from matplotlib import pyplot as plt
import numpy as np
import cv2


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
    :param tgt_sdf:
    :param grid_spacing:
    :param pad:
    :return:
    """

    # compute the march maps
    src_map_h, src_map_w = march_map(src_sdf)

    # move each sampled point to the surface using the march map
    src_h_c = src_map_h[h_i, w_i] + h
    src_w_c = src_map_w[h_i, w_i] + w

    # stack the results in an N, (src,tgt), (h,w) matrix
    return np.stack([src_h_c, src_w_c], axis=1)


fig = plt.figure(figsize=(18, 10))
axes = fig.subplots(1, 2)
t0_plot, t1_plot = axes
fig.show()

episode = np.load('episode.npy')

# list of sample indices
h_i, w_i = grid_sample(*episode.shape[1:3], grid_spacing=16, pad=2)

for t in range(episode.shape[0]-1):

    t0 = episode[t]
    t1 = episode[t + 1]

    [ax.clear() for ax in axes]
    t0_plot.imshow(t0)
    t1_plot.imshow(t1)

    corresp_l = extract_kp(t0, h_i, w_i)

    # plot the corresponding points
    for i in range(corresp_l.shape[0]):
        t0_plot.scatter(corresp_l[i, 0, 1], corresp_l[i, 0, 0])
        t1_plot.scatter(corresp_l[i, 1, 1], corresp_l[i, 1, 0])

    fig.canvas.draw()

