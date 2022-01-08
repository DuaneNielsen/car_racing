from jax import numpy as jnp
import numpy as np
from breadboard import bilinear_interpolate, M
from matplotlib import pyplot as plt


def test_breadboard():
    ax = plt.subplot()
    x_i, y = jnp.meshgrid(jnp.arange(20), jnp.arange(40))
    x_i += 0.5
    map = np.zeros((100, 100))
    for theta in jnp.linspace(0, jnp.pi, 30):
        ax.clear()
        m = M(jnp.array([50, 50, theta]))
        map = bilinear_interpolate(map, x_i, m)
        ax.imshow(map)
        plt.pause(0.05)
    plt.show()
