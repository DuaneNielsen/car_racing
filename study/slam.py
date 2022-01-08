import jax.numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from generate_pose import Episode
from geometry import transform_points

if __name__ == '__main__':
    ep = Episode(4)
    world_fig, world_ax = plt.subplots()

    x, y = [], {}
    for pose in ep.pose:
        v = transform_points(pose, ep.vertices)
        world_ax.add_patch(Polygon(v.T, color='blue', fill=False))
        x.append(v[0, 0])

    plt.show()