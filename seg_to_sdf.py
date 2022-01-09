import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm
import argparse
from PIL import Image
from PIL.ImageOps import grayscale


parser = argparse.ArgumentParser()
parser.add_argument('-ep', '--episode')
parser.add_argument('-v', '--visualize', action='store_true', default=False)
parser.add_argument('-dry', '--dry_run', action='store_true', default=False)
args = parser.parse_args()

fig = plt.figure()
spec = fig.add_gridspec(1, 4)
ax_road = fig.add_subplot(spec[0, 0])
ax_state = fig.add_subplot(spec[0, 1])
ax_segment = fig.add_subplot(spec[0, 2])
ax_sdf = fig.add_subplot(spec[0, 3])


if False:
    ax_road_mask = fig.add_subplot(spec[1, 0])
    ax_state_mask = fig.add_subplot(spec[1, 1])
    ax_segment_mask = fig.add_subplot(spec[1, 2])
    ax_sdf_mask = fig.add_subplot(spec[1, 3])
else:
    ax_road_mask = None
    ax_state_mask = None
    ax_segment_mask = None
    ax_sdf_mask = None


fig.show()

episode = 0
state_stack = np.load(f'data/dataset/{args.episode}_state_stack.npz')['arr_0']
sdf_road_stack = np.load(f'data/dataset/{args.episode}_sdf_road_stack.npz')['arr_0']
state_stack_mask = np.load(f'data/dataset/{args.episode}_state_mask_stack.npz')['arr_0']
sdf_road_stack_mask = np.load(f'data/dataset/{args.episode}_sdf_road_mask_stack.npz')['arr_0']
segment_stack = np.load(f'data/dataset/{args.episode}_segment_stack.npz')['arr_0']
segment_stack_mask = np.load(f'data/dataset/{args.episode}_segment_mask_stack.npz')['arr_0']

segment_thresh_stack = []
sdf2_stack = []


def update(ax, img):
    if ax is not None:
        ax.clear()
        ax.imshow(img, origin='lower')


def gradient(segment):
    # Scharr the edges
    gradient_x = cv2.Sobel(segment, 3, dx=1, dy=0)
    gradient_y = cv2.Sobel(segment, 3, dx=0, dy=1)
    return cv2.addWeighted(gradient_x, 0.5, gradient_y, 0.5, 0.0)


for state, state_mask, road, road_mask, segment, segment_mask in tqdm(zip(state_stack, state_stack_mask, sdf_road_stack, sdf_road_stack_mask, segment_stack, segment_stack_mask)):

    # threshold
    segment = segment > 0.5
    segment_thresh_stack += [segment]

    # Signed Distance Field
    gradient_segment = gradient(segment.astype(float))
    sdf = distance_transform_edt(gradient_segment == 0)
    sign = segment.astype(float) * 2 - 1
    sdf = sdf * sign

    sdf2_stack += [sdf]

    if args.visualize:
        update(ax_state, state.astype(np.uint8))
        update(ax_state_mask, state_mask)
        update(ax_road, road)
        update(ax_road_mask, road_mask)
        update(ax_segment, segment)
        update(ax_segment_mask, segment_mask)
        update(ax_sdf, sdf)
        update(ax_sdf_mask, segment_mask)
        plt.pause(0.0001)

if not args.dry_run:
    for i, segment in enumerate(segment_thresh_stack):
        image = Image.fromarray(segment.astype(np.uint8), mode='L')
        image = grayscale(image)
        image.save(f'data/road_segments/0/{args.episode}_{i}.PNG')

    for i, sdf in enumerate(sdf2_stack):
        np.savez(f'data/road_sdfs/0/{args.episode}_{i}', sdf)