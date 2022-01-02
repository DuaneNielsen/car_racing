import jax.numpy as jnp
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import scipy.interpolate as interpolate
import time

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
        self.N, self.w, self.h = self.sdfs.shape
        self.C = jnp.array([
            [1., 0., 47.],
            [0., 1., 70.],
            [0., 0., 1.]
        ])
        self.verts = jnp.array([[0., self.w, self.w, 0.], [0., 0., self.h, self.h], [1., 1., 1., 1.]])
        self.i = -1
        self.sdf_grid = Grid(self.sdfs.shape[1:])
        self.state_grid = Grid(self.states.shape[1:3])


class MapArray:
    def __init__(self, map_shape):
        self.map = np.zeros(map_shape)
        self.N = np.zeros(map_shape, dtype=int)
        self.shape = map_shape
        self.xlim = (map_shape[0], 0)
        self.ylim = (map_shape[1], 0)
        self.map_grid = Grid(map_shape[0:2])
        self.grid0, self.grid1 = np.meshgrid(np.arange(map_shape[0]), np.arange(map_shape[1]))

    def interpolate(self, pose, image):

        scan_grid = Grid(image.shape[0:2])
        points = scan_grid.index(pose)[0:2].T

        map = interpolate.griddata(
            points=points,
            values=image[scan_grid.grid[0], scan_grid.grid[1]],
            xi=(self.grid0, self.grid1),
            method='linear'
        )

        mask = ~np.isnan(map)
        return map, mask

    def integrate(self, pose_head, image_head, pose_tail, image_tail):

        map, mask = self.interpolate(pose_head, image_head)
        self.map[mask] += map[mask]
        self.N[mask] += 1

        if image_tail is not None:

            map, mask = self.interpolate(pose_tail, image_tail)
            self.map[mask] -= map[mask]
            self.N[mask] -= 1

    def calc_lim(self, mask):
        zmask = ~mask
        if len(mask.shape) > 2:
            zmask = zmask[:, :, 0]
        # get the bounding box by taking the gradient of the mask marginals
        self.xlim = [i.item() for i in np.where(np.convolve(zmask.max(0), np.array([1., 1.])) == 1.)[0]]
        self.ylim = [i.item() for i in np.where(np.convolve(zmask.max(1), np.array([1., 1.])) == 1.)[0]]

    def mean(self):
        mask = self.N == 0
        self.calc_lim(mask)
        the_map = np.zeros(self.shape)
        the_map[~mask] = self.map[~mask] / self.N[~mask]
        return the_map


class Plotter:
    def __init__(self, lim, layout='default'):
        self.lim = lim
        self.map_sdf = MapArray((lim * 2, lim * 2))
        self.map_road = MapArray((lim * 2, lim * 2))
        self.map_state = MapArray((lim * 2, lim * 2, 3))

        self.fig = plt.figure(figsize=(18, 12))

        self.ax_sdf, self.ax_state, self.ax_road = None, None, None
        self.ax_map_sdf, self.ax_map_state, self.ax_map_road = None, None, None

        if layout == 'default':
            gs = self.fig.add_gridspec(9, 4)
            self.ax_sdf = self.fig.add_subplot(gs[0:3, 3])
            self.ax_state = self.fig.add_subplot(gs[3:6, 3])
            self.ax_road = self.fig.add_subplot(gs[6:9, 3])

            self.ax_map_sdf = self.fig.add_subplot(gs[0:3, 0:3])
            self.ax_map_state = self.fig.add_subplot(gs[3:6, 0:3])
            self.ax_map_road = self.fig.add_subplot(gs[6:9, 0:3])

        elif layout == 'sdf_map':
            self.ax_map_sdf = self.fig.add_subplot(1, 1, 1)

        elif layout == 'rgb_map':
            self.ax_map_state = self.fig.add_subplot(1, 1, 1)

        elif layout == 'road':
            self.ax_map_road = self.fig.add_subplot(1, 1, 1)

        self.axes = [[self.ax_sdf, self.ax_state, self.ax_road], [self.ax_map_sdf, self.ax_map_state, self.ax_map_road]]
        self.fig.show()

    def clear(self):
        for row in self.axes:
            for ax in row:
                if ax is not None:
                    ax.clear()

    def draw_maps(self):
        if self.ax_map_sdf is not None:
            self.ax_map_sdf.imshow(self.map_sdf.mean(), origin='lower')
            self.ax_map_sdf.set_aspect('equal')
            self.ax_map_sdf.set_xlim(self.map_sdf.xlim)
            self.ax_map_sdf.set_ylim(self.map_sdf.ylim)

        if self.ax_map_state is not None:
            self.ax_map_state.imshow(np.floor(self.map_state.mean()).astype(np.uint8), origin='lower')
            self.ax_map_state.set_aspect('equal')
            self.ax_map_state.set_xlim(self.map_state.xlim)
            self.ax_map_state.set_ylim(self.map_state.ylim)

        if self.ax_map_road is not None:
            self.ax_map_road.imshow(self.map_road.mean(), origin='lower')
            self.ax_map_road.set_aspect('equal')
            self.ax_map_road.set_xlim(self.map_road.xlim)
            self.ax_map_road.set_ylim(self.map_road.ylim)

    def draw_img(self, ax, img):
        if ax is not None:
            ax.imshow(img, origin='lower')

    def draw_poly(self, ax, verts, color='blue'):
        if ax is not None:
            ax.add_patch(Polygon(verts[0:2].T, color=color, fill=False))

    def draw_points(self, ax, points):
        if ax is not None:
            ax.scatter(points[0], points[1])

    def update(self):
        self.fig.canvas.draw()


if __name__ == '__main__':
    plot = Plotter(lim=1000, layout='default')
    assert jnp.allclose(jnp.eye(3),
                        jnp.matmul(
                            M(jnp.array([1., 1., jnp.pi / 8])),
                            inv(jnp.array([1., 1., jnp.pi / 8]))
                        ),
                        atol=1e-6)

    trj = VehicleTrajectoryObs(12, start=70)
    start = invert_M(trj.pose[0])
    ep_len = trj.sdfs.shape[0]
    window_size = 64

    for t in range(ep_len):

        t_tail, t_head = t-window_size, t

        sdf, sdf_road, state, pose = trj.sdfs[t_head], trj.sdfs_road[t_head], trj.states[t_head], trj.pose[t_head]

        # center pose in map
        pose = jnp.matmul(start, pose)
        pose = jnp.matmul(M(jnp.array([plot.lim / 2, plot.lim / 2, 0.])), pose)

        if t_tail >= 0:
            sdf_tail, sdf_road_tail, state_tail, pose_tail = trj.sdfs[t_tail], trj.sdfs_road[t_tail], trj.states[t_tail], trj.pose[t_tail]
            # center pose in map
            pose_tail = jnp.matmul(start, pose_tail)
            pose_tail = jnp.matmul(M(jnp.array([plot.lim / 2, plot.lim / 2, 0.])), pose_tail)
        else:
            sdf_tail, sdf_road_tail, state_tail, pose_tail = None, None, None, None

        # write to map
        plot.map_sdf.integrate(pose, sdf, pose_tail, sdf_tail)
        plot.map_road.integrate(pose, sdf_road, pose_tail, sdf_road_tail)
        plot.map_state.integrate(pose, state, pose_tail, state_tail)

        if t_tail >= 0:

            plot.clear()

            # show images
            plot.draw_img(plot.ax_sdf, sdf_tail)
            plot.draw_img(plot.ax_state, state_tail)
            plot.draw_img(plot.ax_road, sdf_road_tail)

            # the pose data is in h, w space, but we want the verts in x, y space
            # so we must transpose the co-ordinates
            # transpose_axes = jnp.array([
            #     [0, 1., 0],
            #     [1., 0, 0],
            #     [0, 0, 1.]
            # ])
            # pose_tail = np.matmul(transpose_axes, pose_tail)

            # draw the bounding box and vehicle position
            plot.draw_poly(plot.ax_map_sdf, np.matmul(pose_tail, trj.verts))
            plot.draw_points(plot.ax_map_sdf, np.matmul(np.matmul(pose_tail, trj.C), np.array([[0.], [0.], [1.]])))
            plot.draw_poly(plot.ax_map_state, np.matmul(pose_tail, trj.verts))
            plot.draw_points(plot.ax_map_state, np.matmul(np.matmul(pose_tail, trj.C), np.array([[0.], [0.], [1.]])))
            plot.draw_maps()
            plot.update()

    plt.show()
