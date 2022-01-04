import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
spec = fig.add_gridspec(2, 2)
ax_road = fig.add_subplot(spec[0, 0])
ax_state = fig.add_subplot(spec[0, 1])
ax_road_mask = fig.add_subplot(spec[1, 0])
ax_state_mask = fig.add_subplot(spec[1, 1])

fig.show()

episode = 12
state_stack = np.load(f'data/dataset/{episode}_state_stack.npy')
sdf_road_stack = np.load(f'data/dataset/{episode}_sdf_road_stack.npy')
state_stack_mask = np.load(f'data/dataset/{episode}_state_mask_stack.npy')
sdf_road_stack_mask = np.load(f'data/dataset/{episode}_sdf_road_mask_stack.npy')


for state, state_mask, road, road_mask in zip(state_stack, state_stack_mask, sdf_road_stack, sdf_road_stack_mask):
    ax_road.clear(), ax_state.clear()
    ax_state.imshow(state, origin='lower')
    ax_road.imshow(road, origin='lower')
    ax_state_mask.imshow(state_mask, origin='lower')
    ax_road_mask.imshow(road_mask, origin='lower')
    plt.pause(0.0001)