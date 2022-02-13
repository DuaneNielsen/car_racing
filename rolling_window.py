import numpy as jnp
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import scipy.interpolate as interpolate
from tqdm import tqdm
import cv2
import argparse
from statistics import mean


def R(t):
    return jnp.array([
        [jnp.cos(t[2]), -jnp.sin(t[2])],
        [jnp.sin(t[2]), jnp.cos(t[2])]
    ])


def M(t):
    return jnp.array([
        [jnp.cos(t[2]), -jnp.sin(t[2]), t[0]],
        [jnp.sin(t[2]), jnp.cos(t[2]), t[1]],
        [0., 0., 1.]
    ])


def inv(t):
    xy = jnp.matmul(-R(t).T, t[0:2])
    return jnp.array([
        [jnp.cos(t[2]), jnp.sin(t[2]), xy[0].item()],
        [-jnp.sin(t[2]), jnp.cos(t[2]), xy[1].item()],
        [0., 0., 1.]
    ])


def invert_M(M):
    xy = jnp.matmul(-M[0:2, 0:2].T, M[0:2, 2])
    return jnp.array([
        [M[0, 0].item(), M[1, 0].item(), xy[0].item()],
        [M[0, 1].item(), M[1, 1].item(), xy[1].item()],
        [0., 0., 1.]
    ])


def get_epsilon(dtype: jnp.dtype) -> float:
    return {
        jnp.dtype("float32"): 1e-5,
        jnp.dtype("float64"): 1e-10,
    }[dtype]


def compute_co_effs(theta):
    if theta < get_epsilon(theta.dtype):
        # see http://ethaneade.com/lie.pdf eqn: (130)
        return 1. - theta ** 2 / 6.0, theta / 2.0 - theta ** 3 / 24.0
    else:
        return jnp.sin(theta) / theta, (1 - jnp.cos(theta)) / theta


def exp(t):
    A = jnp.eye(2)
    B = jnp.array([
        [0., -1],
        [1., 0]
    ])
    sin_theta_div_theta, one_minus_cos_div_theta = compute_co_effs(t[2])
    V = sin_theta_div_theta * A + one_minus_cos_div_theta * B
    xy = jnp.matmul(V, t[0:2])
    return jnp.concatenate((xy, t[2:]))


def log(t):
    A = jnp.eye(2)
    B = jnp.array([
        [0., -1],
        [1., 0]
    ])
    sin_theta_div_theta, one_minus_cos_div_theta = compute_co_effs(t[2])
    det = sin_theta_div_theta ** 2 + one_minus_cos_div_theta ** 2
    V = (sin_theta_div_theta * A + one_minus_cos_div_theta * B.T) / det
    xy = jnp.matmul(V, t[0:2])
    return jnp.concatenate((xy, t[2:]))


class Grid:
    def __init__(self, shape):
        self.shape = shape
        grid_0, grid_1 = jnp.meshgrid(*[jnp.arange(d) for d in shape])
        self.grid = grid_0.flatten(), grid_1.flatten()
        self.grid_homo = jnp.stack([*self.grid, jnp.ones_like(self.grid[0])])

    def index(self, M=None):
        if M is None:
            return self.grid_homo
        else:
            return jnp.matmul(M, self.grid_homo)


class VehicleTrajectoryObs:
    def __init__(self, i, start=0):
        """

        :param image: sdf image in h, w
        :param center_xy: center in xy co-ords (h = y and w = x)
        """
        super().__init__()
        self.sdfs_road = jnp.load(f'data/ep{i}_sdf_road.npy')[start:].transpose(0, 2, 1)
        self.sdfs = jnp.load(f'data/ep{i}_sdf.npy')[start:].transpose(0, 2, 1)
        self.states = jnp.load(f'data/ep{i}_state.npy')[start:, :70].transpose(0, 2, 1, 3)
        self.pose = jnp.load(f'data/ep{i}_pose.npy')[start:]
        self.pose_d = jnp.load(f'data/ep{i}_pose_d.npy')[start:]
        self.map = jnp.load(f'data/ep{i}_map.npy')
        self.rms = jnp.load(f'data/ep{i}_error.npy')[start:]
        self.segment = jnp.load(f'data/ep{i}_segment.npy')[start:].astype(float).transpose(0, 2, 1)
        self.N, self.w, self.h = self.sdfs.shape
        self.verts = jnp.array([[0., self.w, self.w, 0.], [0., 0., self.h, self.h], [1., 1., 1., 1.]])
        self.i = -1
        self.sdf_grid = Grid(self.sdfs.shape[1:])
        self.state_grid = Grid(self.states.shape[1:3])

    def get_step(self, t):
        return trj.sdfs[t], trj.sdfs_road[t], trj.states[t], trj.segment[t], trj.pose[t], trj.rms[t]


def interpolate_to_grid(pose, image, dest_grid):

    image_grid = Grid(image.shape[0:2])
    points = image_grid.index(pose)[0:2].T

    map = interpolate.griddata(
        points=points,
        values=image[image_grid.grid[0], image_grid.grid[1]],
        xi=dest_grid,
        method='linear'
    )

    mask = ~np.isnan(map)
    return map, mask


class MapArray:
    def __init__(self, map_shape):
        self.map = np.zeros(map_shape)
        self.N = np.zeros(map_shape, dtype=int)
        self.shape = map_shape
        self.xlim = (map_shape[0], 0)
        self.ylim = (map_shape[1], 0)
        self.map_grid = Grid(map_shape[0:2])
        self.grid0, self.grid1 = np.meshgrid(np.arange(map_shape[0]), np.arange(map_shape[1]))

    def integrate(self, pose_head, image_head, pose_tail=None, image_tail=None):

        map, mask = interpolate_to_grid(pose_head, image_head, (self.grid0, self.grid1))
        self.map[mask] += map[mask]
        self.N[mask] += 1

        if image_tail is not None:

            map, mask = interpolate_to_grid(pose_tail, image_tail, (self.grid0, self.grid1))
            self.map[mask] -= map[mask]
            self.N[mask] -= 1

    def mean(self):
        mask = self.N != 0
        the_map = np.zeros(self.shape)
        the_map[mask] = self.map[mask] / self.N[mask]
        return the_map, mask


class Plotter:
    def __init__(self, lim, layout='default'):
        self.fig = plt.figure(figsize=(18, 12))

        self.ax_sdf, self.ax_state, self.ax_road, self.ax_segment = None, None, None, None
        self.ax_map_sdf, self.ax_map_state, self.ax_map_road, self.ax_map_segment = None, None, None, None
        self.ax_map_sdf_mask, self.ax_map_state_mask, self.ax_map_road_mask, self.ax_map_segment_mask = None, None, None, None

        if layout == 'default':
            gs = self.fig.add_gridspec(15, 7)
            self.ax_sdf = self.fig.add_subplot(gs[0:3, 6])
            self.ax_state = self.fig.add_subplot(gs[3:6, 6])
            self.ax_road = self.fig.add_subplot(gs[6:9, 6])
            self.ax_segment = self.fig.add_subplot(gs[9:12, 6])

            self.ax_map_sdf = self.fig.add_subplot(gs[0:3, 3:6])
            self.ax_map_state = self.fig.add_subplot(gs[3:6, 3:6])
            self.ax_map_road = self.fig.add_subplot(gs[6:9, 3:6])
            self.ax_map_segment = self.fig.add_subplot(gs[9:12, 3:6])

            self.ax_map_sdf_mask = self.fig.add_subplot(gs[0:3, 0:3])
            self.ax_map_state_mask = self.fig.add_subplot(gs[3:6, 0:3])
            self.ax_map_road_mask = self.fig.add_subplot(gs[6:9, 0:3])
            self.ax_map_segment_mask = self.fig.add_subplot(gs[9:12, 0:3])

            self.ax_max_error = self.fig.add_subplot(gs[12:15, :])

        elif layout == 'error':
            self.ax_max_error = self.fig.add_subplot(1, 1, 1)

        elif layout == 'sdf_map':
            self.ax_map_sdf = self.fig.add_subplot(1, 1, 1)

        elif layout == 'rgb_map':
            self.ax_map_state = self.fig.add_subplot(1, 1, 1)

        elif layout == 'road':
            self.ax_map_road = self.fig.add_subplot(1, 1, 1)

        elif layout == 'segment':
            self.ax_map_segment = self.fig.add_subplot(1, 1, 1)

        self.axes = [[self.ax_sdf, self.ax_state, self.ax_road], [self.ax_map_sdf, self.ax_map_state, self.ax_map_road]]
        self.fig.show()

    def clear(self):
        for row in self.axes:
            for ax in row:
                if ax is not None:
                    ax.clear()

    def draw_map(self, ax, map, xlim=None, ylim=None):
        if ax is not None:
            ax.imshow(map, origin='lower')
            ax.set_aspect('equal')
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

    def draw_img(self, ax, img):
        if ax is not None:
            ax.imshow(img, origin='lower')

    def draw_poly(self, ax, verts, color='blue'):
        if ax is not None:
            ax.add_patch(Polygon(verts[0:2].T, color=color, fill=False))

    def draw_points(self, ax, points):
        if ax is not None:
            ax.scatter(points[0], points[1])

    def update_error(self, max_error_stack, min_error_stack, mean_error_stack):
        if self.ax_max_error is not None:
            self.ax_max_error.clear()
            x = list(range(len(max_error_stack)))
            self.ax_max_error.plot(x, max_error_stack)
            self.ax_max_error.plot(x, min_error_stack)
            self.ax_max_error.plot(x, mean_error_stack)

    def update(self):
        self.fig.canvas.draw()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='create egocentric maps')
    parser.add_argument("-ep", "--episode", type=int, default=0)
    parser.add_argument('-v', "--visualize", action='store_true', default=False)
    parser.add_argument('-lb', "--look_behind", type=int, default=128)
    parser.add_argument('-la', "--look_ahead", type=int, default=128)
    parser.add_argument('-ml', "--max_length", type=int, default=np.inf)
    parser.add_argument('-l', "--layout", type=str, default='default')
    args = parser.parse_args()

    episode = args.episode
    lookbehind = args.look_behind
    lookahead = args.look_ahead
    start_step = 70

    m = M(jnp.array([1., 1., 0]))

    plot = Plotter(lim=1000, layout=args.layout)
    trj = VehicleTrajectoryObs(episode, start=start_step)
    start = invert_M(trj.pose[0])
    ep_len = min(args.max_length, trj.sdfs.shape[0])

    window_size = lookahead + lookbehind
    lim = 1000
    map_sdf = MapArray((lim * 2, lim * 2))
    map_road = MapArray((lim * 2, lim * 2))
    map_state = MapArray((lim * 2, lim * 2, 3))
    map_segment = MapArray((lim * 2, lim * 2))

    def to_world(pose):
        pose = jnp.matmul(start, pose)
        return jnp.matmul(M(jnp.array([lim / 2, lim / 2, 0.])), pose)

    sdf_stack, sdf_road_stack, state_stack, sdf_mask_stack, sdf_road_mask_stack, state_mask_stack = [], [], [], [], [], []
    segment_stack, segment_mask_stack, rms_stack = [], [], []
    max_error_stack, min_error_stack, mean_error_stack = [], [], []

    for t in tqdm(range(ep_len)):

        t_tail, t_middle, t_head = t-window_size, t-lookahead, t

        sdf, sdf_road, state, segment, pose, rms = trj.get_step(t_head)
        rms_stack.append(rms)

        pose = to_world(pose)

        if t_tail >= 0:
            sdf_tail, sdf_road_tail, state_tail, segment_tail, pose_tail, _ = trj.get_step(t_tail)
            pose_tail = to_world(pose_tail)

            sdf_middle, sdf_road_middle, state_middle, segment_middle, pose_middle, _ = trj.get_step(t_middle)
            pose_middle = to_world(pose_middle)
            rms_stack.pop(0)

        else:
            sdf_tail, sdf_road_tail, state_tail, segment_tail, pose_tail = None, None, None, None, None
            sdf_middle, sdf_road_middle, state_middle, segment_middle, pose_middle = None, None, None, None, None

        max_error_stack.append(max(rms_stack))
        min_error_stack.append(min(rms_stack))
        mean_error_stack.append(mean(rms_stack))

        # write to map
        map_sdf.integrate(pose, sdf, pose_tail, sdf_tail)
        map_road.integrate(pose, sdf_road, pose_tail, sdf_road_tail)
        map_state.integrate(pose, state, pose_tail, state_tail)
        map_segment.integrate(pose, segment, pose_tail, segment_tail)

        def warp(image, mask, pose, shape):
            return cv2.warpAffine(image, pose[0:2, :], shape), cv2.warpAffine(mask.astype(float), pose[0:2, :], shape)

        if t_tail >= 0:

            centered_size = 220
            # centered_pose = np.matmul(trj.C, pose_tail)

            pose_middle_origin = np.matmul(jnp.eye(3), invert_M(pose_middle))

            def center(pose):
                flip = np.array([
                    [1., 0., 0.],
                    [0., -1., 0.],
                    [0., 0., 1.]
                ])
                centered_pose = np.matmul(flip, pose)
                center_vehicle = np.array([
                    [1., 0., -40.],
                    [0., 1., 20.],
                    [0., 0., 1.]
                ])

                centered_pose = np.matmul(center_vehicle, centered_pose)
                centered_pose = np.matmul(M(jnp.array([centered_size / 2, centered_size / 2, 0.])), centered_pose)
                return centered_pose

            centered_pose = center(pose_middle_origin)
            target_shape = (centered_size, centered_size)
            centered_sdf, centered_sdf_mask = warp(*map_sdf.mean(), centered_pose, target_shape)
            centered_road, centered_road_mask = warp(*map_road.mean(), centered_pose, target_shape)
            centered_state, centered_state_mask = map_state.mean()
            centered_state = np.floor(centered_state).astype(np.uint8)
            centered_state, centered_state_mask = warp(centered_state, centered_state_mask, centered_pose, target_shape)
            centered_segment, centered_segment_mask = warp(*map_segment.mean(), centered_pose, target_shape)

            sdf_stack += [centered_sdf]
            sdf_road_stack += [centered_road]
            state_stack += [centered_state]
            segment_stack += [centered_segment]

            sdf_mask_stack += [centered_sdf_mask]
            sdf_road_mask_stack += [centered_road_mask]
            state_mask_stack += [centered_state_mask]
            segment_mask_stack += [centered_segment_mask]

            if args.visualize:
                plot.clear()

                # show images
                plot.draw_img(plot.ax_sdf, sdf_middle)
                plot.draw_img(plot.ax_state, state_middle)
                plot.draw_img(plot.ax_road, sdf_road_middle)
                plot.draw_img(plot.ax_segment, segment_middle)

                # draw maps
                plot.draw_map(plot.ax_map_sdf, centered_sdf)
                plot.draw_map(plot.ax_map_road, centered_road)
                plot.draw_map(plot.ax_map_state, centered_state)
                plot.draw_map(plot.ax_map_segment, centered_segment)

                plot.draw_map(plot.ax_map_sdf_mask, centered_sdf_mask)
                plot.draw_map(plot.ax_map_road_mask, centered_road_mask)
                plot.draw_map(plot.ax_map_state_mask, centered_state_mask)
                plot.draw_map(plot.ax_map_state_mask, centered_state_mask)
                plot.draw_map(plot.ax_map_segment_mask, centered_segment_mask)
                plot.update_error(max_error_stack, min_error_stack, mean_error_stack)

                C = jnp.array([
                    [1., 0., 47.],
                    [0., 1., 70.],
                    [0., 0., 1.]
                ])

                # draw the bounding box and vehicle position
                center_t = center(np.eye(3))
                plot.draw_poly(plot.ax_map_sdf, np.matmul(center_t, trj.verts))
                plot.draw_points(plot.ax_map_sdf, np.matmul(np.matmul(center_t, C), np.array([[0.], [0.], [1.]])))
                plot.draw_poly(plot.ax_map_state, np.matmul(center_t, trj.verts))
                plot.draw_points(plot.ax_map_state, np.matmul(np.matmul(center_t, C), np.array([[0.], [0.], [1.]])))

                plot.update()

    # save
    def save(filename, stack):
        np.savez(f'data/dataset/{episode}_{filename}', np.stack(stack))

    save('sdf_stack', sdf_stack)
    save('sdf_road_stack', sdf_road_stack)
    save('state_stack', state_stack)
    save('sdf_mask_stack', sdf_mask_stack)
    save('sdf_road_mask_stack', sdf_road_mask_stack)
    save('state_mask_stack', state_mask_stack)
    save('segment_stack', segment_stack)
    save('segment_mask_stack', segment_mask_stack)
    save('max_error_stack', max_error_stack)


