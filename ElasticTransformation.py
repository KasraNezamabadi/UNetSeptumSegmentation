import numpy as np
import cv2
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


def elastic_transform(image, mask, alpha, sigma, random_state=None):

    assert len(image.shape)==2
    assert len(mask.shape)==2

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1))

    result_mask = map_coordinates(mask, indices, order=1).reshape(shape)
    _, result_mask = cv2.threshold(src=result_mask, thresh=128, maxval=255, type=cv2.THRESH_BINARY)

    return map_coordinates(image, indices, order=1).reshape(shape), result_mask
