import jax.numpy as jnp
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon


class SE2:
    def __init__(self, t=None):
        if t is not None:
            self._t = t
        else:
            self._t = jnp.zeros(3)

    @staticmethod
    def from_xytheta(x=0., y=0., theta=0.):
        t = SE2()
        t._t = jnp.array([x, y, theta])
        return t

    def as_tuple(self):
        return self._t[0], self._t[1], self._t[2]

    @property
    def x(self):
        return self._t[0]

    @property
    def y(self):
        return self._t[1]

    @property
    def pos(self):
        return self._t[0:2]

    @property
    def pos_homo(self):
        return jnp.concatenate((self._t[0:2], jnp.ones(1)))

    @property
    def theta(self):
        return self._t[2]

    @property
    def R(self):
        return jnp.array([
            [jnp.cos(self.theta), -jnp.sin(self.theta)],
            [jnp.sin(self.theta), jnp.cos(self.theta)]
        ])

    @property
    def M(self):
        return jnp.array([
            [jnp.cos(self.theta), -jnp.sin(self.theta), self.x],
            [jnp.sin(self.theta), jnp.cos(self.theta), self.y],
            [0., 0., 1.]
        ])

    @property
    def inv(self):
        xy = jnp.matmul(-self.R.T, self.pos)
        return jnp.array([
            [jnp.cos(self.theta), jnp.sin(self.theta), xy[0].item()],
            [-jnp.sin(self.theta), jnp.cos(self.theta), xy[1].item()],
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


def exp(se2):
    A = jnp.eye(2)
    B = jnp.array([
        [0., -1],
        [1., 0]
    ])
    sin_theta_div_theta, one_minus_cos_div_theta = compute_co_effs(se2[2])
    V = sin_theta_div_theta * A + one_minus_cos_div_theta * B
    xy = jnp.matmul(V, se2[0:2])
    return jnp.concatenate((xy, se2[2:]))


def log(SE2):
    A = jnp.eye(2)
    B = jnp.array([
        [0., -1],
        [1., 0]
    ])
    sin_theta_div_theta, one_minus_cos_div_theta = compute_co_effs(SE2[2])
    det = sin_theta_div_theta ** 2 + one_minus_cos_div_theta ** 2
    V = (sin_theta_div_theta * A + one_minus_cos_div_theta * B.T) / det
    xy = jnp.matmul(V, SE2[0:2])
    return jnp.concatenate((xy, SE2[2:]))


class Frame:
    def __init__(self):
        self.world_t = SE2()


class SDFImage(Frame):
    def __init__(self, image):
        """

        :param image: sdf image in h, w
        :param center_xy: center in xy co-ords (h = y and w = x)
        """
        super().__init__()
        self.image = image.T
        w, h = self.image.shape[0], self.image.shape[1]
        self.h = h
        self.w = w
        self._verts = jnp.array([[-w/2., w/2, w/2, -w/2], [-h/2., -h/2., h/2, h/2], [1., 1., 1., 1.]])

    @property
    def verts(self):
        return jnp.matmul(self.world_t.M, self._verts)


class Plotter:
    def __init__(self, lim):
        self.fig = plt.figure(figsize=(12, 18))
        self.ax = self.fig.subplots(1, 1)
        self.fig.show()
        self.lim = lim

    def clear(self):
        self.ax.clear()

    def draw_poly(self, verts, color='blue'):
        self.ax.add_patch(Polygon(verts[0:2].T, color=color, fill=False))

    def draw_points(self, points):
        self.ax.scatter(points[0], points[1])

    def update(self):
        self.ax.set_aspect('equal')
        self.ax.set_xlim((-self.lim, self.lim))
        self.ax.set_ylim((-self.lim, self.lim))
        self.fig.canvas.draw()


if __name__ == '__main__':
    plot = Plotter(lim=200.)
    assert jnp.allclose(jnp.eye(3),
                        jnp.matmul(
                            SE2.from_xytheta(1., 1., jnp.pi / 8).M,
                            SE2.from_xytheta(1., 1., jnp.pi / 8).inv
                        ),
                        atol=1e-7)

    image = jnp.ones((15, 20))
    images = [SDFImage(image=image) for _ in range(20)]
    lin = lambda m, x, c: m * x + c
    vs = [jnp.array([lin(-5, i, 100.), lin(10, i, -80.), lin(jnp.pi/20., i, -jnp.pi/5)]) for i in range(20)]
    rest = []
    for v, s, in zip(vs, images):
        for t in jnp.linspace(0., 1., 20):
            plot.clear()
            s.world_t = SE2(exp(v * t))
            plot.draw_poly(s.verts)
            for poly in rest:
                plot.draw_poly(poly)
            plot.update()
        rest += [s.verts]

    plt.show()

