from matplotlib import pyplot as plt
import numpy as np
import cv2
import geometry as geo


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


fig = plt.figure(figsize=(18, 10))
axes = fig.subplots(1, 2)
t0_plot, t1_plot = axes
fig.show()

episode = np.load('episode.npy')

# list of sample indices
grid = geo.grid_sample(*episode.shape[1:3], grid_spacing=16, pad=2)

for t in range(30, episode.shape[0]-1):

    t0 = episode[t]
    t1 = episode[t + 1]

    [ax.clear() for ax in axes]
    t0_plot.imshow(t0)
    t1_plot.imshow(t1)

    t0_kp = extract_kp(t0, grid[1], grid[0])
    t1_kp = extract_kp(t1, grid[1], grid[0])

    # plot the corresponding points
    for i in range(t0_kp.shape[1]):
        t0_plot.scatter(t0_kp[1, i], t0_kp[0, i])
        t1_plot.scatter(t1_kp[1, i], t1_kp[0, i])

    fig.canvas.draw()
    plt.show()

