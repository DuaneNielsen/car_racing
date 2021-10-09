import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import geometry as geo


if __name__ == '__main__':

    fig = plt.figure()
    r1, r2, world = fig.subplots(1, 3)
    fig.show()

    f1 = geo.Scan(20, 30)
    assert np.allclose(np.matmul(f1.M, f1.inv_M), np.eye(3))

    f2 = geo.Scan(20, 30, x=5, y=8, theta=np.radians(30))
    assert np.allclose(np.matmul(f2.inv_M, f2.M), np.eye(3))

    grid = geo.grid_sample(f1.h, f1.w, grid_spacing=4, pad=2)
    M = np.matmul(f1.inv_M, f2.M)
    grid_f2 = geo.transform_points(M, grid.T).T

    r1.imshow(f1.i, origin='lower')
    r2.imshow(f2.i, origin='lower')
    for i in range(len(grid)):
        r1.scatter(grid[i, 0], grid[i, 1])
        r2.scatter(grid_f2[i, 0], grid_f2[i, 1])
    r1.set_aspect('equal')
    r2.set_aspect('equal')

    wf1 = geo.transform_points(f1.M, f1.vertices.T)
    wf2 = geo.transform_points(f2.M, f2.vertices.T)
    inside = geo.inside(grid.T, wf2)
    wf1 = Polygon(wf1.T, color=[1, 0, 0], fill=False)
    wf2 = Polygon(wf2.T, color=[0, 1, 0], fill=False)
    world.add_patch(wf1)
    world.add_patch(wf2)
    world.scatter(grid[inside, 0], grid[inside, 1], color='blue')
    world.scatter(grid[~inside, 0], grid[~inside, 1], color='red')
    world.set_aspect('equal')
    world.autoscale_view()

    plt.show()