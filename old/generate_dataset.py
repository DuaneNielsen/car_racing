import jax.numpy as jnp
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from tqdm import tqdm
import jax

cpu0 = jax.devices('cpu')[0]
gpu0 = jax.devices('gpu')[0]


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
        self.states = jnp.load(f'data/ep{i}_state.npy')[start:, :86].transpose(0, 2, 1, 3)
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


def bilinear_interpolate(map, sdf, M):
    shape = sdf.shape[0:2]
    grid_0, grid_1 = jnp.meshgrid(*[jnp.arange(d) for d in shape])
    grid_0, grid_1 = grid_0.flatten(), grid_1.flatten()
    source_index = jnp.stack([grid_0, grid_1, jnp.ones_like(grid_0)])
    dest_index = jnp.matmul(M, source_index)[0:2]
    rem = jnp.mod(dest_index, 1)
    dest_index = dest_index.astype(int)
    rem_x = rem[0]
    rem_y = rem[1]
    sdf = jnp.pad(sdf, mode='edge', pad_width=1)[1:, 1:]
    floor_sdf = sdf[source_index[0], source_index[1]]
    x_sdf = sdf[source_index[0] + 1, source_index[1]]
    y_sdf = sdf[source_index[0], source_index[1] + 1]
    map[dest_index[0], dest_index[1]] = floor_sdf * (1 - rem_x) + x_sdf * rem_x + floor_sdf * (1 - rem_y) + y_sdf * rem_y
    return map


class MapArray:
    def __init__(self, shape):
        self.map = jnp.zeros(shape)
        self.N = jnp.zeros(shape, dtype=int)
        self.xlim = (shape[0], 0)
        self.ylim = (shape[1], 0)

    def update(self, dest_index, source_index, sdf):
        self.xlim = min(self.xlim[0], dest_index[1].min()), max(self.xlim[1], dest_index[1].max())
        self.ylim = min(self.ylim[0], dest_index[0].min()), max(self.ylim[1], dest_index[0].max())
        N = self.N[dest_index[0], dest_index[1]]
        self.map[dest_index[0], dest_index[1]] = (self.map[dest_index[0], dest_index[1]] * N + sdf[source_index[0], source_index[1]]) / (N + 1)
        self.N[dest_index[0], dest_index[1]] += 1



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

    def draw_map(self, ax_map, map, xlim=None, ylim=None):
        ax_map.imshow(map, origin='lower')
        ax_map.set_aspect('equal')
        if xlim is not None:
            ax_map.set_xlim(xlim)
        if ylim is not None:
            ax_map.set_ylim(ylim)

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


class BoundingBoxes:
    def __init__(self, N, dims=2):
        self.min, self.max = jnp.full((dims, N), jnp.inf, jnp.uint64), jnp.full((dims, N), -jnp.inf, jnp.uint64)
        self.N = N
        self.dims = dims

    def update(self, i, grid_indexes):
        for d, grid_index in enumerate(grid_indexes):
            self.min[d, i] = grid_index.min()
            self.max[d, i] = grid_index.max()

    def lims(self, start=0, end=None):
        end = self.N if end is None else end
        a_lims = []
        for d in range(self.dims):
            a_lims += [(self.min[d, start:end].min(), self.max[d, start:end].max())]
        return a_lims


if __name__ == '__main__':
    plot = Plotter(lim=1000, layout='default')
    assert jnp.allclose(jnp.eye(3),
                        jnp.matmul(
                            M(jnp.array([1., 1., jnp.pi / 8])),
                            inv(jnp.array([1., 1., jnp.pi / 8]))
                        ),
                        atol=1e-6)

    start = 70
    end = 170
    ep = 12

    sdf_road = jnp.load(f'data/ep{ep}_sdf_road.npy')[start:end].transpose(0, 2, 1)
    sdf_road = jax.device_put(sdf_road, cpu0)
    #sdf = jnp.load(f'data/ep{ep}_sdf.npy')[start:end].transpose(0, 2, 1)
    #state = jnp.load(f'data/ep{ep}_state.npy')[start:end, :86].transpose(0, 2, 1, 3)
    poses = jnp.load(f'data/ep{ep}_pose.npy')[start:end]
    # pose_d = jnp.load(f'data/ep{ep}_pose_d.npy')[start:]
    # map = jnp.load(f'data/ep{ep}_map.npy')
    # rms = jnp.load(f'data/ep{ep}_error.npy')[start:]

    l, w, h = sdf_road.shape
    sdf_grid = Grid(sdf_road.shape[1:])
    #state_grid = Grid(state.shape[1:3])

    # sdf_map = jnp.zeros((l, 2000, 2000), dtype=jnp.float32)
    # sdf_mask = jnp.zeros((l, 2000, 2000), dtype=bool)
    # sdf_bb = BoundingBoxes(l)

    sdf_road_map = jnp.zeros((l, 800, 800), dtype=jnp.float32)
    sdf_road_mask = jnp.zeros((l, 800, 800), dtype=bool)
    sdf_road_bb = BoundingBoxes(l)

    # state_map = jnp.zeros((l, 2000, 2000, 3), dtype=jnp.float32)
    # state_mask = jnp.zeros((l, 2000, 2000, 3), dtype=bool)
    # state_bb = BoundingBoxes(l)

    start = invert_M(poses[0])

    for i in tqdm(range(l)):

        # center pose in map
        pose = jnp.matmul(start, poses[i])
        pose = jnp.matmul(M(jnp.array([plot.lim / 2, plot.lim / 2, 0.])), pose)

        # write sdf_road map
        # dest_index, source_index = sdf_grid.index(pose), sdf_grid.index()
        # sdf_map[i, dest_index[0], dest_index[1]] = sdf[i, source_index[0], source_index[1]]
        # sdf_mask[i, dest_index[0], dest_index[1]] = True
        # sdf_bb.update(i, grid_indexes=dest_index[0:2])

        # write sdf_road map
        dest_index, source_index = sdf_grid.index(pose), sdf_grid.index()
        sdf_road_map[i, dest_index[0], dest_index[1]] = sdf_road[i, source_index[0], source_index[1]]
        sdf_road_mask[i, dest_index[0], dest_index[1]] = True
        sdf_road_bb.update(i, grid_indexes=dest_index[0:2])

        # write state map
        # dest_index, source_index = state_grid.index(pose), state_grid.index()
        # state_map[i, dest_index[0], dest_index[1]] = state[i, source_index[0], source_index[1]]
        # state_mask[i, dest_index[0], dest_index[1]] = True
        # state_bb.update(i, grid_indexes=dest_index[0:2])

    # setup the size of the sensor window
    verts = jnp.array([[0., w, w, 0.], [0., 0., h, h], [1., 1., 1., 1.]])
    C = jnp.array([
        [1., 0., 47.],
        [0., 1., 70.],
        [0., 0., 1.]
    ])
    transpose_axes = jnp.array([
        [0, 1., 0],
        [1., 0, 0],
        [0, 0, 1.]
    ])

    window_size = 256

    # plot the results
    for i in range(l-window_size):

        plot.clear()

        # show images
        # plot.draw_img(plot.ax_sdf, sdf[i])
        # plot.draw_img(plot.ax_state, state[i])
        plot.draw_img(plot.ax_road, sdf_road[i])

        # compute and draw sdf map for range
        # sdf_map_instance = sdf_map[i:i+window_size].sum(axis=0) / (sdf_mask[i:i+window_size].sum(axis=0) + 1)
        # plot.draw_map(plot.ax_map_sdf, sdf_map_instance, *reversed(sdf_bb.lims(end=i+1)))

        # compute and draw sdf map for range
        sdf_road_map_instance = sdf_road_map[i:i+window_size].sum(axis=0) / (sdf_road_mask[i:i+window_size].sum(axis=0) + 1)
        plot.draw_map(plot.ax_map_road, sdf_road_map_instance, *reversed(sdf_road_bb.lims(end=i+1)))

        # compute and draw state for range
        # state_instance = (state_map[i:i+window_size].sum(axis=0) / (state_mask[i:i+window_size].sum(axis=0) + 1)).astype(np.uint8)
        # plot.draw_map(plot.ax_map_state, state_instance, *reversed(state_bb.lims(end=i+1)))

        # the pose data is in h, w space, but we want the verts in x, y space
        # so we must transpose the co-ordinates
        # center pose in map
        pose = jnp.matmul(start, poses[i])
        pose = jnp.matmul(M(jnp.array([plot.lim / 2, plot.lim / 2, 0.])), pose)
        pose = jnp.matmul(transpose_axes, pose)

        # draw the bounding box and vehicle position
        plot.draw_poly(plot.ax_map_sdf, jnp.matmul(pose, verts))
        plot.draw_points(plot.ax_map_sdf, jnp.matmul(jnp.matmul(pose, C), jnp.array([[0.], [0.], [1.]])))

        # plot.draw_poly(plot.ax_map_state, jnp.matmul(pose, verts))
        # plot.draw_points(plot.ax_map_state, jnp.matmul(jnp.matmul(pose, C), jnp.array([[0.], [0.], [1.]])))
        plot.update()

    plt.show()
