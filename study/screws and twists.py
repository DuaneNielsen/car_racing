import numpy as np
import jax.numpy as jnp
from jaxlie import SE2
from matplotlib import pyplot as plt

if __name__ == '__main__':

    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.set_xlim(-3.0, 3.0)
    ax.set_ylim(-3.0, 3.0)

    screw = jnp.array([-1., 0., 0.5])
    i_pos = jnp.array([
        [0., 1.],
        [0., 0.]
    ]).T

    for t in jnp.linspace(0, jnp.pi, 50):
        M = SE2.exp(screw * t)
        pos = M.apply(i_pos)
        ax.scatter(pos[0], pos[1])

plt.show()