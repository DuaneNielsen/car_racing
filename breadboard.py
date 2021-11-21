import jax.numpy as jnp
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import numpy as np


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
        grid_0, grid_1 = grid_0.flatten(), grid_1.flatten()
        self.grid = jnp.stack([grid_0, grid_1, jnp.ones_like(grid_0)])

    def index(self, M=None):
        if M is None:
            return self.grid
        else:
            return jnp.matmul(M, self.grid).astype(jnp.int32)


class VehicleTrajectoryObs:
    def __init__(self, i, start=0):
        """

        :param image: sdf image in h, w
        :param center_xy: center in xy co-ords (h = y and w = x)
        """
        super().__init__()
        self.sdfs_road = jnp.load(f'data/ep{i}_sdf_road.npy')[start:].transpose(0, 2, 1)
        self.sdfs = jnp.load(f'data/ep{i}_sdf.npy')[start:].transpose(0, 2, 1)
        self.states = jnp.load(f'data/ep{i}_state.npy')[start:, :60].transpose(0, 2, 1, 3)
        self.pose = jnp.load(f'data/ep{i}_pose.npy')[start:]
        self.pose_d = jnp.load(f'data/ep{i}_pose_d.npy')[start:]
        self.map = jnp.load(f'data/ep{i}_map.npy')
        self.rms = jnp.load(f'data/ep{i}_error.npy')[start:]
        self.N, self.w, self.h = self.sdfs.shape
        self.C = jnp.array([
            [1., 0., 48.],
            [0., 1., 86.],
            [0., 0., 1.]
        ])
        self.verts = jnp.array([[0., self.w, self.w, 0.], [0., 0., self.h, self.h], [1., 1., 1., 1.]])
        self.i = -1
        self.grid = Grid(self.sdfs.shape[1:])

    def __getitem__(self, item):
        return self.sdfs[item], self.states[item], self.pose[item]

    def __len__(self):
        return self.N

    def __iter__(self):
        self.i = -1
        return self

    def __next__(self):
        self.i += 1
        if self.i >= self.N:
            raise StopIteration()
        return self[self.i]


class Plotter:
    def __init__(self, lim):
        self.fig = plt.figure(figsize=(12, 18))
        gs = self.fig.add_gridspec(6, 4)
        self.ax_sdf, self.ax_state = self.fig.add_subplot(gs[0:3, 3]), self.fig.add_subplot(gs[3:6, 3])
        self.ax_map_sdf, self.ax_map_state = self.fig.add_subplot(gs[0:3, 0:3]), self.fig.add_subplot(gs[3:6, 0:3])
        self.axes = [[self.ax_sdf, self.ax_state], [self.ax_map_sdf, self.ax_map_state]]
        self.fig.show()
        self.lim = lim
        self.map = np.zeros((lim * 2, lim * 2))
        self.map_state = np.zeros((lim * 2, lim * 2, 3), dtype=np.uint8)

    def clear(self):
        for row in self.axes:
            for ax in row:
                ax.clear()

    def draw_img(self, ax, img):
        ax.imshow(img, origin='lower')

    def draw_maps(self):
        self.ax_map_sdf.imshow(self.map, origin='lower')
        self.ax_map_state.imshow(self.map_state, origin='lower')

    def draw_poly(self, ax, verts, color='blue'):
        ax.add_patch(Polygon(verts[0:2].T, color=color, fill=False))

    def draw_points(self, ax, points):
        ax.scatter(points[0], points[1])

    def update(self):

        for ax in self.axes[1]:
            ax.set_aspect('equal')
            ax.set_xlim((0, self.lim))
            ax.set_ylim((0, self.lim))
        self.fig.canvas.draw()


if __name__ == '__main__':
    plot = Plotter(lim=1000)
    assert jnp.allclose(jnp.eye(3),
                        jnp.matmul(
                            M(jnp.array([1., 1., jnp.pi / 8])),
                            inv(jnp.array([1., 1., jnp.pi / 8]))
                        ),
                        atol=1e-7)

    trj = VehicleTrajectoryObs(8, start=500)
    start = invert_M(trj.pose[0])

    for sdf, state, pose in trj:
        plot.clear()

        # show images
        plot.draw_img(plot.ax_sdf, sdf)
        plot.draw_img(plot.ax_state, state)

        # center pose in map
        pose = jnp.matmul(start, pose)
        pose = jnp.matmul(M(jnp.array([plot.lim / 2, plot.lim / 2, 0.])), pose)

        # write to map
        dest_index, source_index = trj.grid.index(pose), trj.grid.index()
        plot.map[dest_index[0], dest_index[1]] = sdf[source_index[0], source_index[1]]
        plot.map_state[dest_index[0], dest_index[1]] = state[source_index[0], source_index[1]]

        # the pose data is in h, w space, but we want the verts in x, y space
        # so we must transpose the co-ordinates
        transpose_axes = jnp.array([
            [0, 1., 0],
            [1., 0, 0],
            [0, 0, 1.]
        ])
        pose = np.matmul(transpose_axes, pose)

        # draw the bounding box and vehicle position
        plot.draw_poly(plot.ax_map_sdf, np.matmul(pose, trj.verts))
        plot.draw_points(plot.ax_map_sdf, np.matmul(np.matmul(pose, trj.C), np.array([[0.], [0.], [1.]])))
        plot.draw_poly(plot.ax_map_state, np.matmul(pose, trj.verts))
        plot.draw_points(plot.ax_map_state, np.matmul(np.matmul(pose, trj.C), np.array([[0.], [0.], [1.]])))
        plot.draw_maps()
        plot.update()

    plt.show()
