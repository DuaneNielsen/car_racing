import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


def road_segment(observation):
    return np.logical_and(observation[:, :, 1] > 100, observation[:, :, 1] < 110)


def gradient(segment):
    gradient_x = cv2.Sobel(segment, 3, dx=1, dy=0)
    gradient_y = cv2.Sobel(segment, 3, dx=0, dy=1)
    return cv2.addWeighted(gradient_x, 0.5, gradient_y, 0.5, 0.0)


def road_distance_field(observation, segment_func):
    segment = segment_func(observation)
    grad = gradient(segment.astype(float))
    df = distance_transform_edt(grad == 0)
    sign = segment * 2 - 1
    return df * sign, sign