import numpy as np
import jax.numpy as jnp
from jaxlie import SE2
from matplotlib import pyplot as plt

twist = np.array([1.0, 0.0, 0.05])
T_w_b = SE2.exp(twist)
print(T_w_b.rotation())
print(T_w_b.translation())
print(T_w_b.parameters())


def forward_kinematics_dh(theta1, theta2, theta3):
    T01 = SE2.from_xy_theta(0.0, 0, theta1)
    T12 = SE2.from_xy_theta(1.0, 0, theta2)
    T23 = SE2.from_xy_theta(0.5, 0, theta3)
    T34 = SE2.from_xy_theta(1.0, 0, 0)
    J1 = T01
    J2 = T01.multiply(T12)
    J3 = T01.multiply(T12.multiply(T23))
    J4 = T01.multiply(T12.multiply(T23.multiply(T34)))
    return J1, J2, J3, J4


if __name__ == '__main__':
    fig, ax = plt.subplots()

    for t in jnp.linspace(0, jnp.pi):
        joints = forward_kinematics_dh(t, t, t)
        ax.clear()
        ax.set_xlim(5.0, -5.0)
        ax.set_ylim(5.0, -5.0)
        for j, color in zip(joints, ['red', 'green', 'blue', 'magenta']):
            ax.scatter(j.translation()[0], j.translation()[1], color=color)
        plt.pause(0.05)