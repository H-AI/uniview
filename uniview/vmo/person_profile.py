# IICO/VMO -- see IT, feel IT, love IT
# Copyright 2020 The IICO Corporate. All Rights Reserved.
#
# ============================================================================
""" COCO Person Drawing Utilities """
from typing import Union

import cv2
import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont
from dataclasses import dataclass

from .keypoints import keypoint_array
from .constants import vmo_skeleton, torso_skeleton
from .bbox import UniviewBBox

try:
    FONT = ImageFont.truetype("arial.ttf", 24)
except IOError:
    FONT = ImageFont.load_default()

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

CocoColors = [
    [255, 0, 0],
    [255, 85, 0],
    [255, 170, 0],
    [255, 255, 0],
    [170, 255, 0],
    [85, 255, 0],
    [0, 255, 0],
    [0, 255, 85],
    [0, 255, 170],
    [0, 255, 255],
    [0, 170, 255],
    [0, 85, 255],
    [0, 0, 255],
    [85, 0, 255],
    [170, 0, 255],
    [255, 0, 255],
    [255, 0, 170],
    [255, 0, 85],
    [255, 0, 0],
]


def compute_color_for_labels(label):
    """ palette by class label """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def _draw_single_box(
    image,
    xmin,
    ymin,
    xmax,
    ymax,
    color=(0, 255, 0),
    display_str=None,
    font=None,
    width=2,
    alpha=0.5,
    fill=False,
):
    if font is None:
        font = FONT

    draw = ImageDraw.Draw(image, mode="RGBA")
    left, right, top, bottom = xmin, xmax, ymin, ymax
    alpha_color = color + (int(255 * alpha),)
    draw.rectangle(
        [(left, top), (right, bottom)],
        outline=color,
        fill=alpha_color if fill else None,
        width=width,
    )

    if display_str:
        text_bottom = bottom
        # Reverse list and print from bottom to top.
        text_width, text_height = font.getsize(display_str)
        margin = np.ceil(0.05 * text_height)
        draw.rectangle(
            xy=[
                (left + width, text_bottom - text_height - 2 * margin - width),
                (left + text_width + width, text_bottom - width),
            ],
            fill=alpha_color,
        )
        draw.text(
            (
                left + margin + width,
                text_bottom - text_height - margin - width,
            ),
            display_str,
            fill="black",
            font=font,
        )

    return image


def draw_boxes(
    image: np.ndarray,
    boxes: Union[list, np.ndarray],
    labels: Union[list, np.ndarray] = None,
    scores: Union[list, np.ndarray] = None,
    class_name_map: Union[list, dict] = None,
    line_width: int = 2,
    alpha: float = 0.5,
    fill: bool = False,
    font=None,
    score_format=":{:.2f}",
):
    """Draw bboxes(labels, scores) on image
    Args:
        image: numpy array image, shape should be (height, width, channel)
        boxes: bboxes, shape should be (N, 4), and each row is
               (xmin, ymin, xmax, ymax) in ratios
        labels: labels, shape: (N, )
        scores: label scores, shape: (N, )
        class_name_map: list or dict, map class id to class name for
                        visualization.
        line_width: box line width
        alpha: text background alpha
        fill: fill box or not
        font: text font
        score_format: score format
    Returns:
        An image with information drawn on it.
    """
    imh, imw, _ = image.shape
    boxes = np.array(boxes)
    boxes *= np.array([imw, imh, imw, imh])
    num_boxes = boxes.shape[0]
    if isinstance(image, Image.Image):
        draw_image = image
    elif isinstance(image, np.ndarray):
        draw_image = Image.fromarray(image)
    else:
        raise AttributeError("Unsupported images type {}".format(type(image)))

    for i in range(num_boxes):
        display_str = ""
        color = (0, 255, 0)
        if labels is not None:
            this_class = labels[i]
            color = compute_color_for_labels(this_class)
            class_name = (
                class_name_map[this_class]
                if class_name_map is not None
                else str(this_class)
            )
            display_str = class_name

        if scores is not None:
            prob = scores[i]
            if display_str:
                display_str += score_format.format(prob)
            else:
                display_str += "score" + score_format.format(prob)

        draw_image = _draw_single_box(
            image=draw_image,
            xmin=boxes[i, 0],
            ymin=boxes[i, 1],
            xmax=boxes[i, 2],
            ymax=boxes[i, 3],
            color=color,
            display_str=display_str,
            font=font,
            width=line_width,
            alpha=alpha,
            fill=fill,
        )

    image = np.array(draw_image, dtype=np.uint8)
    return image


def find_contours(*args, **kwargs):
    """
    Wraps cv2.findContours to maintain compatibility between versions 3 and 4
    Returns:
        contours, hierarchy
    """
    if cv2.__version__.startswith("4"):
        contours, hierarchy = cv2.findContours(*args, **kwargs)
    elif cv2.__version__.startswith("3"):
        _, contours, hierarchy = cv2.findContours(*args, **kwargs)
    else:
        raise AssertionError(
            "cv2 must be either version 3 or 4 to call this method"
        )
    return contours, hierarchy


def draw_masks(
    image,
    masks,
    labels=None,
    border=True,
    border_width=2,
    border_color=(255, 255, 255),
    alpha=0.5,
    color=None,
):
    """
    Args:
        image: numpy array image, shape should be (height, width, channel)
        masks: (N, 1, Height, Width)
        labels: mask label
        border: draw border on mask
        border_width: border width
        border_color: border color
        alpha: mask alpha
        color: mask color
    Returns:
        np.ndarray
    """
    if isinstance(image, Image.Image):
        image = np.array(image)
    assert isinstance(image, np.ndarray)
    masks = np.array(masks)
    for i, mask in enumerate(masks):
        mask = mask.squeeze()[:, :, None].astype(np.bool)

        label = labels[i] if labels is not None else 1
        _color = (
            compute_color_for_labels(label) if color is None else tuple(color)
        )

        image = np.where(
            mask, mask * np.array(_color) * alpha + image * (1 - alpha), image
        )
        if border:
            contours, hierarchy = find_contours(
                mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            image = cv2.drawContours(
                image,
                contours,
                -1,
                border_color,
                thickness=border_width,
                lineType=cv2.LINE_AA,
            )

    image = image.astype(np.uint8)
    return image


def vis_keypoints(
    image: np.ndarray,
    keypoints: Union[list, np.ndarray],
    diameter: int = 5,
    show_label: bool = True,
) -> np.ndarray:
    """ Draw person keypoints. Only valid and visible keypoints will be shown
    Args:
        image: numpy array image, shape should be (height, width, channel)
        keypoints: The keypoints input can be in many different formats:
                   1). list of long list, each of the sub-list is a long
                       repetitive [x, y, v, ...]
                   2). list of list of keypoints, each of the sub-list is a
                       list of (x, y, v) keypoints.
                   3). ndarray of shape [n_instance, n_keypoints, 3]
                   4). list of ndarray of shape [n_keypoints, 3]
                   5). list of list of tuples in (x, y)
        diameter: radius of keypoint circle
        show_label: mark keypoint label next to joint
    Returns:
        np.ndarray
    """
    image = image.copy()
    keypoints = keypoint_array(keypoints)
    if len(keypoints.shape) == 3:
        n_inst, n_kps, _ = keypoints.shape
        for i in range(n_inst):
            for j in range(n_kps):
                if keypoints.shape[2] == 3:
                    x, y, v = keypoints[i, j, :]
                    if v < 2:
                        continue
                if keypoints.shape[2] == 2:
                    x, y = keypoints[i, j, :]
                    if x < 0 or y < 0:
                        continue
                cv2.circle(
                    image, (int(x), int(y)), diameter, CocoColors[j], -1
                )
                if show_label:
                    cv2.putText(
                        image,
                        str(j),
                        (int(x) + 5, int(y)),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        0.7,
                        (255, 255, 255),
                    )
    return image


def viz_vmo_annotation(
    image: np.ndarray,
    bboxes: Union[list, np.ndarray] = None,
    keypoints: Union[list, np.ndarray] = None,
    labels: Union[list, np.ndarray] = None,
    box_format: str = "albu",
    resize: float = 1.0,
) -> np.ndarray:
    """Visualize vmo example data with person annotations

    Args:
        image: np.ndarray, image data in RGB format
        box_format: one of ['coco', 'yolo', 'albu']
        bboxes: list or np.ndarray
        categories: np.ndarray,
        keypoints: np.ndarray,
        r_resize: if not equals to 1.0, resize image to target ratio

    Retruns:
        display image
    """
    im = np.copy(image[:, :, ::-1])
    if bboxes is not None:
        if not box_format == "albu":
            bboxes = UniviewBBox().get_albu_bbox(bboxes, box_format, im.shape)
        im = draw_boxes(im, bboxes, labels)
        im = vis_keypoints(im, keypoints)
    if resize != 1.0:
        im = cv2.resize(im, (0, 0), fx=resize, fy=resize)
    return im


def viz_vmo_pose(
    vmo_example: dataclass, show_box: bool = True, r_resize: float = 1.0
) -> np.ndarray:
    """Visualize vmo example data with person skeletions

    Args:
        vmo_example: a dataclass object that has the following
        necessary fields:
        - imdata: np.ndarray, image data in RGB format
        - bboxes: np.ndarray, bounding boxes in albumentation format
        - categories: np.ndarray,
        - keypoints: np.ndarray,
        - keypoint_format: str, keypoint format in ['vmo', 'torso']
        show_box: if True, display bounding box and category label
        r_resize: if not equals to 1.0, resize image to target ratio

    Retruns:
        display image
    """
    if vmo_example.keypoint_format not in ["vmo", "torso"]:
        raise Exception("//Error: please provide valid vmo keypoint format!")
    if vmo_example.keypoint_format == "vmo":
        _skeleton = vmo_skeleton
    elif vmo_example.keypoint_format == "torso":
        _skeleton = torso_skeleton
    else:
        raise Exception("//Error: undefined skeleton format!")

    imdsp = np.copy(vmo_example.imdata[:, :, ::-1])
    if show_box:
        imdsp = draw_boxes(imdsp, vmo_example.bboxes, vmo_example.categories)
    imdsp = vis_keypoints(imdsp, vmo_example.keypoints)

    kps_foramt = None
    if vmo_example.keypoints.shape[2] == 3:
        kps_foramt = "xyv"
    elif vmo_example.keypoints.shape[2] == 2:
        kps_foramt = "xy"
    else:
        raise Exception("//Error: unknown keypoint format!")
    for _inst in vmo_example.keypoints:
        for p, q in _skeleton:
            if kps_foramt == "xyv":
                x0, y0, v0 = _inst[p]
                x1, y1, v1 = _inst[q]
                if v0 < 2 or v1 < 2:
                    continue
            if kps_foramt == "xy":
                x0, y0 = _inst[p]
                x1, y1 = _inst[q]
                if x0 < 0 or y0 < 0 or x1 < 0 or y1 < 0:
                    continue
            cv2.line(
                imdsp, (int(x0), int(y0)), (int(x1), int(y1)), CocoColors[p], 2
            )

    if r_resize != 1.0:
        imdsp = cv2.resize(imdsp, (0, 0), fx=r_resize, fy=r_resize)

    return imdsp
