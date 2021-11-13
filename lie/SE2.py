import jax.numpy as np


def from_xytheta(x=0.0, y=0.0, theta=0.0):
    return np.array([x, y, theta])


def tangent_matrix(self):
    return np.array([
        [0., -self.theta, self.x],
        [self.theta, 0, self.y],
        [0., 0., 0.]
    ])


def transform_matrix(SE2):
    return np.array([
        [np.cos(SE2[2]), -np.sin(SE2[2]), SE2[0]],
        [np.sin(SE2[2]), np.cos(SE2[2]), SE2[1]],
        [0., 0., 1.]
    ])


def adjoint_matrix(SE2):
    return np.array([
        [np.cos(SE2[2]), -np.sin(SE2[2]), SE2[1]],
        [np.sin(SE2[2]), np.cos(SE2[2]), -SE2[0]],
        [0., 0., 1.]
    ])


def get_epsilon(dtype: np.dtype) -> float:
    return {
        np.dtype("float32"): 1e-5,
        np.dtype("float64"): 1e-10,
    }[dtype]


def compute_co_effs(theta):
    if theta < get_epsilon(theta.dtype):
        # see http://ethaneade.com/lie.pdf eqn: (130)
        return 1. - theta ** 2 / 6.0, theta / 2.0 - theta ** 3 / 24.0
    else:
        return np.sin(theta) / theta, (1 - np.cos(theta))/theta


def exp(se2):
    A = np.eye(2)
    B = np.array([
        [0., -1],
        [1., 0]
    ])
    sin_theta_div_theta, one_minus_cos_div_theta = compute_co_effs(se2[2])
    V = sin_theta_div_theta * A + one_minus_cos_div_theta * B
    xy = np.matmul(V, se2[0:2])
    return np.concatenate((xy, se2[2:]))


def log(SE2):
    A = np.eye(2)
    B = np.array([
        [0., -1],
        [1., 0]
    ])
    sin_theta_div_theta, one_minus_cos_div_theta = compute_co_effs(SE2[2])
    det = sin_theta_div_theta ** 2 + one_minus_cos_div_theta ** 2
    V = (sin_theta_div_theta * A + one_minus_cos_div_theta * B.T) / det
    xy = np.matmul(V, SE2[0:2])
    return np.concatenate((xy, SE2[2:]))