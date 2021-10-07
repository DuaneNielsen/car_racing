import gym
from matplotlib import pyplot as plt
import cv2
import numpy as np
from numpy.linalg import svd
from scipy.ndimage import distance_transform_edt


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


def extract_keypoints(sdf, h_index, w_index):
    """
    computes a set of corresponding points for two sdfs by grid sampling
    and following the sdf gradient to the nearest surface

    this should work well when sensors are running a rate much faster than
    the robot angular or translational speed

    :param sdf: sdf dimensions h, w
    :param grid_spacing: spacing of sample grid
    :param pad: padding of sample area
    :return: list of keypoints
    """

    # compute the march maps
    src_map_h, src_map_w = march_map(sdf)

    # move each sampled point to the surface using the march map
    src_h_c = src_map_h[h, w] + h
    src_w_c = src_map_w[h, w] + w

    # stack the results in an (h,w), N matrix
    kp = np.stack([src_h_c, src_w_c], axis=0)
    return kp


def rms(source, target):
    return np.sum((source - target) ** 2) / source.shape[1]


def icp(source, target):
    """

    :param source: set of source points, D, N
    :param target: set of target points, D, N
    :return: aligned target points, Rotation matrix, translation
    """
    source_center = source.mean(axis=1, keepdims=True)
    target_center = target.mean(axis=1, keepdims=True)
    t = - target_center + source_center
    target = target + t
    corr = np.matmul(source, target.T)
    u, d, v = svd(corr)
    R = np.matmul(u, v.T)
    target = np.matmul(R, target)
    return target, R, t


def to_sdf(state):
    # crop the input image
    crop = state[:62]

    # segment the image
    segment = np.logical_and(crop[:, :, 1] > 100, crop[:, :, 1] < 110).astype(float)

    # sobel the edges
    gradient_x = cv2.Sobel(segment, 3, dx=1, dy=0)
    gradient_y = cv2.Sobel(segment, 3, dx=0, dy=1)
    gradient = cv2.addWeighted(gradient_x, 0.5, gradient_y, 0.5, 0.0)

    # Distance Field
    return distance_transform_edt(gradient == 0)


fig = plt.figure(figsize=(18, 10))
axes = fig.subplots(1, 4)
raw_plot, sdf_plot, pose_plot, trajectory_plot = axes
fig.show()

env = gym.make('CarRacing-v0')
prev_state, done = env.reset(), False
prev_sdf = to_sdf(prev_state)
action = env.action_space.sample()
env.render()

episode = []
step = 0

# list of sample indices
h, w = grid_sample(*prev_sdf.shape, grid_spacing=16, pad=4)

while not done:
    state, reward, done, info = env.step(action)
    action = env.action_space.sample()

    # wait for zoom in to complete before starting forward
    if step > 30:
        action[1], action[2] = 0.5, 0.0  # gas, brake
    else:
        action[1], action[2] = 0.0, 1.0

    sdf = to_sdf(state)

    sdf_kp = extract_keypoints(sdf, h, w)
    prev_sdf_kp = extract_keypoints(prev_sdf, h, w)
    i = 0
    while rms(sdf_kp, prev_sdf_kp) > 1.0:
        source, R, t = icp(corr_list[0], corr_list[1])
        i += 1
        if i > 7:
            break

    print(rms(source, corr_list[1]))

    [ax.clear() for ax in axes]
    raw_plot.imshow(state)
    sdf_plot.imshow(sdf)
    sdf_plot.scatter(source[1], source[0])
    vector = np.array([1, 0.]).reshape(2, 1)
    vector += t
    vector = np.matmul(R, vector)
    vector = np.stack([np.zeros_like(vector), vector], axis=1)
    pose_plot.set_xlim(-3, 3)
    pose_plot.set_ylim(-3, 3)
    pose_plot.plot(vector[0], vector[1])

    #env.render()
    fig.canvas.draw()
    if len(episode) > 300:
        break

    prev_sdf = sdf
    step += 1


#np.save('episode', episode)