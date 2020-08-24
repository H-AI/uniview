# IICO/VMO -- see IT, feel IT, love IT
# Copyright 2020 The IICO Corporate. All Rights Reserved.
#
# ============================================================================
""" Test 2D Bounding Box Functions """

import numpy as np

from uniview.IM2D.bbox import yolo2coco, coco2albu, yolo2albu


def test_yolo2coco():

    bboxes = [[0.1, 0.8, 0.4, 0.5], [0.9, 0.8, 0.4, 0.5]]
    tgt_bboxes = [[0.0, 55.0, 40.0, 44.0], [70.0, 55.0, 29.0, 44.0]]
    imshape = (100.0, 100.0)  # (height, width, detph)

    bboxes = np.array(bboxes)
    nbb = yolo2coco(bboxes, imshape)

    assert np.amin(nbb) >= 0.0
    assert np.allclose(nbb, np.array(tgt_bboxes))


def test_coco2albu():

    bboxes = [[-10.0, 55.0, 40.0, 144.0], [70.0, 55.0, 150.0, 60.0]]
    tgt_bboxes = [[0.0, 0.55, 0.15, 0.99], [0.35, 0.55, 0.995, 0.99]]
    imshape = (100, 200)  # (height, width, detph)

    bboxes = np.array(bboxes)
    nbb = coco2albu(bboxes, imshape)

    assert np.amin(nbb) >= 0.0
    assert np.allclose(nbb, np.array(tgt_bboxes))


def test_yolo2albu():

    bboxes = [[0.1, 0.8, 0.4, 0.5], [0.9, 0.8, 0.4, 0.5]]
    tgt_bboxes = [[0.0, 0.55, 0.3, 0.99], [0.7, 0.55, 0.995, 0.99]]
    imshape = (100, 200)  # (height, width, detph)

    bboxes = np.array(bboxes)
    nbb = yolo2albu(bboxes, imshape)

    assert np.amin(nbb) >= 0.0
    assert np.allclose(nbb, np.array(tgt_bboxes))
