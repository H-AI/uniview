# IICO/VMO -- see IT, feel IT, love IT
# Copyright 2020 The IICO Corporate. All Rights Reserved.
#
# ============================================================================
""" Detection 2D Drawing Utilities """
from typing import Any, Union

import numpy as np
import PIL.Image as Image
import PIL.ImageDraw as ImageDraw
import PIL.ImageFont as ImageFont

try:
    FONT = ImageFont.truetype("arial.ttf", 24)
except IOError:
    FONT = ImageFont.load_default()


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label):
    """palette by class label"""
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
    if isinstance(font, int):
        font = ImageFont.truetype("arial.ttf", font)
    elif font is None:
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
    font: Union[Any, int] = None,
    score_format: str = ":{:.2f}",
    box_in_ratios: bool = False,
) -> np.ndarray:
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
        box_in_ratios: if True, box values are in ratios w.r.t width and height
    Returns:
        An image with information drawn on it.
    """
    imh, imw, _ = image.shape
    boxes = np.array(boxes)
    num_boxes = boxes.shape[0]
    if num_boxes > 0 and box_in_ratios:
        boxes *= np.array([imw, imh, imw, imh])
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
