# IICO/VMO -- see IT, feel IT, love IT
# Copyright 2020 The IICO Corporate. All Rights Reserved.
#
# ============================================================================
""" 2D Bounding Box Utilities """

from typing import Union
import numpy as np


def yolo2coco(
    boxes: np.ndarray, imsize: tuple, trim: bool = True
) -> np.ndarray:
    """ convert from yolo format to coco format
    Args:
        box: shape of `[num_box, 4]`. The 4 box ratios of format cx, cy, w, h
    Returns:
        shape of `[num_box, 4]`. The 4 box coordinates of format x1, y1, w, h
        If trum==True,  Any negative value will be cliped to 0.
                        Width will be limited to inside boundary.
                        Height will be limited to inside boundary.
    """
    boxes_new = boxes.copy()
    boxes_new = boxes_new.astype(float)
    boxes_new[:, 0] -= 0.5 * boxes[:, 2]
    boxes_new[:, 1] -= 0.5 * boxes[:, 3]
    boxes_new *= np.array([imsize[1], imsize[0], imsize[1], imsize[0]])
    if trim:
        boxes_new = boxes_new.clip(min=0.0)
        for i in range(boxes_new.shape[0]):
            boxes_new[i, 2] = min(
                boxes_new[i, 2], imsize[1] - boxes_new[i, 0] - 1
            )
            boxes_new[i, 3] = min(
                boxes_new[i, 3], imsize[0] - boxes_new[i, 1] - 1
            )
    return boxes_new


def coco2albu(
    boxes: np.ndarray, imsize: tuple, trim: bool = True
) -> np.ndarray:
    """ convert from coco format to albumentation box format
    Args:
        box: shape of `[num_box, 4]`. The 4 box coordinates of format
        [x1, y1, w, h]

    Returns:
        shape of `[num_box, 4]`. The box of format [x1, y1, x2, y2]
        in ratios.
        If trum==True,  Any negative value will be cliped to 0.
                        Any out of bound value will be clipped to boundary-1
    """
    if not min(imsize) > 1.0:
        raise ValueError("[im2d.bbox.coco2albu] Image size error!")
    boxes_new = boxes.copy()
    boxes_new = boxes_new.astype(float)
    boxes_new[:, 2] += boxes_new[:, 0]
    boxes_new[:, 3] += boxes_new[:, 1]
    if trim:
        for i in range(boxes_new.shape[0]):
            boxes_new[i, 0] = max(boxes_new[i, 0], 0.0)
            boxes_new[i, 1] = max(boxes_new[i, 1], 0.0)
            boxes_new[i, 2] = min(boxes_new[i, 2], imsize[1] - 1)
            boxes_new[i, 3] = min(boxes_new[i, 3], imsize[0] - 1)
    boxes_new *= np.array(
        [1 / imsize[1], 1 / imsize[0], 1 / imsize[1], 1 / imsize[0]]
    )
    return boxes_new


def yolo2albu(
    boxes: np.ndarray, imsize: tuple, trim: bool = True
) -> np.ndarray:
    """ convert from yolo format to coco format
    Args:
        box: shape of `[num_box, 4]`. The 4 box ratios of format cx, cy, w, h
    Returns:
        shape of `[num_box, 4]`. The box of format [x1, y1, x2, y2]
        in ratios.
        If trum==True,  Any negative value will be cliped to 0.
                        Any out of bound value will be clipped to boundary-1
    """
    boxes_new = boxes.copy()
    boxes_new = boxes_new.astype(float)
    boxes_new[:, 0] -= 0.5 * boxes[:, 2]
    boxes_new[:, 1] -= 0.5 * boxes[:, 3]
    boxes_new[:, 2] = boxes[:, 0] + 0.5 * boxes[:, 2]
    boxes_new[:, 3] = boxes[:, 1] + 0.5 * boxes[:, 3]

    if trim:
        if not min(imsize) > 1.0:
            raise ValueError("[im2d.bbox.yolo2albu] Image size error!")
        boxes_new *= np.array([imsize[1], imsize[0], imsize[1], imsize[0]])
        for i in range(boxes_new.shape[0]):
            boxes_new[i, 0] = max(boxes_new[i, 0], 0.0)
            boxes_new[i, 1] = max(boxes_new[i, 1], 0.0)
            boxes_new[i, 2] = min(boxes_new[i, 2], imsize[1] - 1)
            boxes_new[i, 3] = min(boxes_new[i, 3], imsize[0] - 1)
        boxes_new *= np.array(
            [1 / imsize[1], 1 / imsize[0], 1 / imsize[1], 1 / imsize[0]]
        )
    return boxes_new


class UnviewBBox(object):
    """ Uniview bounding box uses Albumentation bounding box format.
        This class convert bounding boxes objects to albu format.
        The folliwng formats are considered:
        - pascal_voc: [x_min, y_min, x_max, y_max], e.g. [98, 345, 420, 462]
        - albumentations: [x_min, y_min, x_max, y_max],
                          e.g. [0.153125, 0.71875, 0.65625, 0.9625]
        - coco = 2: [x_min, y_min, width, height], e.g. [98, 345, 322, 117]
        - yolo = 3: [x_center, y_center, width, height],
                    e.g. [0.4046875, 0.8614583, 0.503125, 0.24375]
    """

    conversions = {"coco": coco2albu, "yolo": yolo2albu}

    def __init__(
        self,
        need_trim: bool = True,
        bbox: Union[list, np.ndarray] = None,
        in_format: str = None,
        imsize: tuple = None,
    ):
        """
        Args:
            bbox: bounding box instance in list of 4 float values:
                [b0, b1, b2, b3]
            imsize: a tuple in (height, width, depth) or (height, width)
            in_format: input bbox format in one of
                        [pascal_voc, albumentations, coco, yolo]
        """
        self.bbox = None
        self.trim = need_trim
        if bbox:
            self._to_albu(bbox, in_format, imsize)

    def _to_albu(
        self, bbox: Union[list, np.ndarray], in_format: str, imsize: tuple
    ):
        """
        Args:
            bbox: bounding box instance in list of 4 float values:
                [b0, b1, b2, b3]
            imsize: a tuple in (height, width, depth) or (height, width)
            in_format: input bbox format in one of
                        [pascal_voc, albumentations, coco, yolo]
        """
        if in_format is None:
            raise ValueError(
                "[im2d.bbox.UniviewBBox] Need input format has to process!"
            )
        elif in_format not in ["yolo", "coco"]:
            raise ValueError(
                "[im2d.bbox.UniviewBBox] Method not implemented yet!"
            )
        if bbox is None:
            raise ValueError(
                "[im2d.bbox.UniviewBBox] No valid data to process!"
            )
        if not min(imsize) > 1.0:
            raise ValueError("[im2d.bbox.UniviewBBox] Image size error!")
        if isinstance(bbox, list) and not isinstance(bbox[0], (list, tuple)):
            bbox = [bbox]
        if not isinstance(bbox, np.ndarray):
            bbox = np.array(bbox)
        self.bbox = self.conversions[in_format](bbox, imsize, trim=self.trim)

    def get_albu_bbox(
        self, bbox: Union[list, np.ndarray], in_format: str, imsize: tuple
    ):
        self._to_albu(bbox, in_format, imsize)
        return self.get_bbox()

    def get_bbox(self):
        """
        Return:
            numpy array in albumentation bbox format
        """
        return self.bbox
