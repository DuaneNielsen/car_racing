from lie import SO2, SE2
import jax.numpy as np
from matplotlib import pyplot as plt


def test_so2():
    av1 = SO2.so2(np.degrees(30.0))
    av2 = SO2.so2(np.degrees(20.0))
    a1 = SO2.exp(av1 + av2)
    x = np.array([1., 0.]).reshape(2, 1)
    y = np.array([0., 1.]).reshape(2, 1)
    x = np.matmul(a1.as_matrix(), x)
    y = np.matmul(a1.as_matrix(), y)
    assert x[0, 0] == np.cos(np.degrees(50.0))
    assert x[1, 0] == np.sin(np.degrees(50.0))
    assert y[0, 0] == -np.sin(np.degrees(50.0))
    assert y[1, 0] == np.cos(np.degrees(50.0))


x_m, y_m = np.meshgrid(np.arange(-10, 10), np.arange(-10, 10))
x_m, y_m = x_m.flatten(), y_m.flatten()
g = np.stack((x_m, y_m, np.ones_like(x_m)))


def plot_field(T, pause=None):
    g_t = np.matmul(T, g)
    fig, ax = plt.subplots()
    for x, y, x_, y_ in zip(x_m, y_m, g_t[0], g_t[1]):
        ax.quiver(x, y, x_, y_)
    if pause is None:
        plt.show()
    else:
        plt.pause(pause)


def test_SE2_plt_field():

    w = SE2.from_xytheta(x=6.0, y=0, theta=9e-6)
    T = SE2.exp(w)
    dT = SE2.log(T)
    print('')
    print(w)
    print(dT)
    assert(np.allclose(dT, w))

    plot_field(SE2.transform_matrix(T), pause=3.0)

    w = SE2.from_xytheta(x=6.0, y=0, theta=np.pi/4.0)
    T = SE2.exp(w)
    dT = SE2.log(T)
    print('')
    print(w)
    print(dT)
    assert(np.allclose(dT, w, atol=1e-5))

    plot_field(SE2.transform_matrix(dT), pause=3.0)


def test_taylor():
    theta = 9e-6
    sin_div_theta, one_minus_cos_div_theta = SE2.compute_co_effs(np.array([theta], dtype=np.float32))
    print("")
    assert abs(sin_div_theta - np.sin(theta) / theta) < 1e-5
    assert abs(one_minus_cos_div_theta - (1 - np.cos(theta)) / theta) < 1e-5


def test_circle():
    fig, ax = plt.subplots()

    i = np.array([1., 0, 1.])
    a = SE2.from_xytheta(x=0., y=0., theta=1.0)
    path = []

    for t in np.linspace(0., np.pi * 2, 20):
        path += [np.matmul(SE2.transform_matrix(SE2.exp(a * t)), i)]
    path = np.stack(path)

    ax.clear()
    ax.scatter(path[:, 0], path[:, 1])

    plt.pause(3.0)


def test_adjoint():
    fig, (ax_s, ax_b) = plt.subplots(1, 2)

    i = np.array([1., 0, 1.])
    a = SE2.from_xytheta(x=0., y=0., theta=1.0)

    path = []

    for t in np.linspace(0., np.pi * 2, 20):
        path += [np.matmul(SE2.transform_matrix(SE2.exp(a * t)), i)]
    path = np.stack(path)

    ax_b.clear()
    ax_b.scatter(path[:, 0], path[:, 1])

    adj = SE2.adjoint_matrix(SE2.from_xytheta(x=2., y=0., theta=0.))
    path = []
    for t in np.linspace(0., np.pi * 2, 20):
        b = SE2.transform_matrix(SE2.exp(a * t))
        s = np.matmul(adj, b)
        path += [np.matmul(s, i)]
    path = np.stack(path)

    ax_s.clear()
    ax_s.scatter(path[:, 0], path[:, 1])

    plt.show()


def test_inverse():
    x = SE2.from_xytheta(1., 2., 3.)
    i = np.matmul(SE2.inv_transform_matrix(x), SE2.transform_matrix(x))
    assert np.allclose(i, np.eye(3), atol=1e-5)