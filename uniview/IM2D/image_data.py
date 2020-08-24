# IICO/VMO -- see IT, feel IT, love IT
# Copyright 2020 The IICO Corporate. All Rights Reserved.
#
# ==========================================================================
""" Image Data Utilities """

from typing import Union
from numpy import np


def color_image_shape_check(data: Union[list, np.ndarray]):
    """Check the shape of a list of image data

    Arguments:
        data {list/ndarray} -- Each of the element is
            in the shape of  `[height, width, depth]`.
            The `num_channel` should be the same for all elements
            in one of [3, 4]

    Raises:
        ValueError: If any of the input data is not in shape of
        `[height, width, depth]`.
        ValueError: If not all `depth` are the same.
    """
    if isinstance(data, np.ndarray):
        data = [data]
    for datum in data:
        if len(datum.shape) != 3:
            raise ValueError(
                "[color_image_shape_check]: Incorrect image data format!"
            )
        if data[0].shape[2] not in [3, 4]:
            raise ValueError(
                "[color_image_shape_check]: Incorrect image data format!"
            )
        if datum.shape[2] != data[0].shape[2]:
            raise ValueError(
                "[color_image_shape_check]: Inconsistent image depth!"
            )
