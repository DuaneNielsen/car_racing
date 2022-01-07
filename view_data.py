import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
spec = fig.add_gridspec(2, 3)
ax_road = fig.add_subplot(spec[0, 0])
ax_state = fig.add_subplot(spec[0, 1])
ax_segment = fig.add_subplot(spec[0, 2])

ax_road_mask = fig.add_subplot(spec[1, 0])
ax_state_mask = fig.add_subplot(spec[1, 1])
ax_segment_mask = fig.add_subplot(spec[1, 2])

fig.show()

episode = 0
state_stack = np.load(f'data/dataset/{episode}_state_stack.npz')['arr_0']
sdf_road_stack = np.load(f'data/dataset/{episode}_sdf_road_stack.npz')['arr_0']
state_stack_mask = np.load(f'data/dataset/{episode}_state_mask_stack.npz')['arr_0']
sdf_road_stack_mask = np.load(f'data/dataset/{episode}_sdf_road_mask_stack.npz')['arr_0']
segment_stack = np.load(f'data/dataset/{episode}_segment_stack.npz')['arr_0']
segment_stack_mask = np.load(f'data/dataset/{episode}_segment_mask_stack.npz')['arr_0']


def update(ax, img):
    ax.clear()
    ax.imshow(img, origin='lower')


for state, state_mask, road, road_mask, segment, segment_mask in zip(state_stack, state_stack_mask, sdf_road_stack, sdf_road_stack_mask, segment_stack, segment_stack_mask):

    update(ax_state, state.astype(np.uint8))
    update(ax_state_mask, state_mask)
    update(ax_road, road)
    update(ax_road_mask, road_mask)
    update(ax_segment, segment)
    update(ax_segment_mask, segment_mask)
    plt.pause(0.0001)