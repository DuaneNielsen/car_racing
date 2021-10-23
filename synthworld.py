import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import geometry as geo


class World:
    def __init__(self, h=60, w=90, x=0.0, y=0.0, theta=0.0):
        self.x_, self.y_, self.theta_ = x, y, theta
        fig = plt.figure()
        axes = fig.subplots(1, 2)
        self.world, self.model = axes
        self.f = geo.Scan(h, w, x=x, y=y, theta=theta)
        self.grid = geo.grid_sample(500, 500, 10, 0)
        inside = geo.inside(self.grid, geo.transform_points(self.f.M, self.f.vertices))
        self.f.image = geo.transform_points(self.f.inv_M, self.grid[:, inside])

    def reset(self):
        self.f.x = self.x_
        self.f.y = self.y_
        self.f.theta = self.theta_
        return self.f.image

    def render(self):
        self.model.clear(), self.world.clear()
        self.world.scatter(self.grid[0], self.grid[1])
        self.world.add_patch(Polygon(geo.transform_points(self.f.M, self.f.vertices).T, color=[1, 0, 0], fill=False))
        self.model.scatter(*self.f.image)
        plt.pause(0.5)

    def step(self, dx, dy, dtheta):
            M_prev, inv_M_prev = self.f.M, self.f.inv_M
            inside_prev = geo.inside(self.grid, geo.transform_points(M_prev, self.f.vertices))
            self.f.x += dx
            self.f.y += dy
            self.f.theta += dtheta
            inside_curr = geo.inside(self.grid, geo.transform_points(self.f.M, self.f.vertices))
            inside = inside_curr & inside_prev
            image_prev = geo.transform_points(inv_M_prev, self.grid[:, inside])
            self.f.image = geo.transform_points(self.f.inv_M, self.grid[:, inside])
            return image_prev, self.f.image


if __name__ == '__main__':

    world = World(x=200.0, y=200)
    world.reset()
    world.render()
    fig, diag = plt.subplots(1, 1)
    for t in range(200):
        prev, curr = world.step(0.5, 1.5, np.radians(1.0))
        diag.clear()
        diag.scatter(*prev, color='blue')
        diag.scatter(*curr, color='red')
        plt.pause(0.01)
        world.render()