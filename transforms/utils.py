# _*_coding=utf-8 _*_
# @author Luyao Huang
# @date 2021/6/10 下午5:42

import numpy as np


def preserve_channel_dim(func):
    """
    Preserve dummy channel dim.

    """
    def wrapped_function(img, *args, **kwargs):
        shape = img.shape
        result = func(img, *args, **kwargs)
        if len(shape) == 3 and shape[-1] == 1 and len(result.shape) == 2:
            result = np.expand_dims(result, axis=-1)
        return result

    return wrapped_function


def preserve_shape(func):
    """
    Preserve shape of the image

    """
    def wrapped_function(img, *args, **kwargs):
        shape = img.shape
        result = func(img, *args, **kwargs)
        result = result.reshape(shape)
        return result

    return wrapped_function


def is_rgb_image(image):
    return len(image.shape) == 3 and image.shape[-1] == 3


def is_grayscale_image(image):
    return (len(image.shape) == 2) or (len(image.shape) == 3 and image.shape[-1] == 1)


def clip(img, dtype, maxval):
    return np.clip(img, 0, maxval).astype(dtype)