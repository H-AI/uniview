# IICO/VMO -- see IT, feel IT, love IT
# Copyright 2020 The IICO Corporate. All Rights Reserved.
#
# ============================================================================
""" 2D Person Keypoint Utilities """

from typing import Union
import numpy as np


def keypoint_array(keypoints: Union[list, np.ndarray]) -> np.ndarray:
    """Convert keypoints to ndarray

    Args:
        keypoints: The keypoints input can be in many different formats:
                   1). list of long list, each of the sub-list is a long
                       repetitive [x, y, v, ...]
                   2). list of list of keypoints, each of the sub-list is a
                       list of (x, y, v) keypoints.
                   3). ndarray of shape [n_instance, n_keypoints, 3]
                   4). list of ndarray of shape [n_keypoints, 3]
                   5). list of list of tuples in (x, y)

    Returns:
        np.ndarray in the format of [n_instance, n_keypoints, 3], where
        the 3 keypoint data are (x, y, v) with v for quality value.
    """
    keypoints = np.array(keypoints)
    if keypoints.shape[0] > 0:
        if len(keypoints.shape) not in [2, 3]:
            raise Exception("// Error: keypoint data in wrong data shape!")
        if len(keypoints.shape) == 2 and keypoints.shape[1] % 3 == 0:
            keypoints = keypoints.reshape((keypoints.shape[0], -1, 3))
    return keypoints


def coco_2_openpose(coco_kps: Union[list, np.ndarray]) -> np.ndarray:
    """ convert from coco keypoint format to openpose format

    HY: based on the 17 COCO annotated keypoints, here the keypoints are
    remapped to revised openpose sequence.

    Furthermore, there adds a new keypoint, which marks the middle point
    between the 2 shoulder keypoints. This new keypoint is labelled as
    the 2nd keypoint in the new sequence.


    Args:
        coco_kps: array of keypoints in coco format

    Returns:
        array of keypoints in revised openpose-18 format
    """
    coco_kps = keypoint_array(coco_kps)
    pt18_joint_list = []
    transform = list(
        zip(
            [0, 5, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3],
            [0, 6, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15, 2, 1, 4, 3],
        )
    )
    for prev_joint in coco_kps:
        new_joint = []
        for idx1, idx2 in transform:
            j1 = prev_joint[idx1]
            j2 = prev_joint[idx2]
            new_joint.append(
                ((j1[0] + j2[0]) / 2, (j1[1] + j2[1]) / 2, min(j1[2], j2[2]))
            )
        pt18_joint_list.append(new_joint)
    return np.array(pt18_joint_list)


def coco_2_vmo(coco_kps: Union[list, np.ndarray]) -> np.ndarray:
    """ convert from coco keypoint format to vmo-14 format

    VMO has 14 keypoints. Here based on the 17 COCO annotated keypoints,
    the keypoints are remapped with new sequence.
    1). Add one NECK keypoint, which marks the middle point between
        the 2 should keypoints.
    2). Find Head as the center of available (Nose), Eyes, and Ears.

    Args:
        coco_kps: array of keypoints in coco format

    Returns:
        array of keypoints in vmo-14 format
    """
    coco_kps = keypoint_array(coco_kps)
    pt14_joint_list = []
    transform = list(
        zip(
            [0, 5, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15],
            [0, 6, 6, 8, 10, 5, 7, 9, 12, 14, 16, 11, 13, 15],
        )
    )
    for prev_joint in coco_kps:
        new_joint = []
        for idx1, idx2 in transform:
            j1 = prev_joint[idx1]
            j2 = prev_joint[idx2]
            new_joint.append(
                ((j1[0] + j2[0]) / 2, (j1[1] + j2[1]) / 2, min(j1[2], j2[2]))
            )

        # -: reprocess head as the center of nose, eyes, and ears
        head_parts_x = []
        head_parts_y = []
        for i in range(1, 5):  # 0:nose (removed); 1,2: eyes; 3,4: ears in coco
            jt = prev_joint[i]
            if jt[2] >= 2:
                head_parts_x.append(jt[0])
                head_parts_y.append(jt[1])

        if head_parts_x and head_parts_y:
            new_joint[0] = (
                int(sum(head_parts_x) / float(len(head_parts_x))),
                int(sum(head_parts_y) / float(len(head_parts_y))),
                2,
            )
        else:
            new_joint[0] = (-1000, -1000, 0)

        # -: add skeleton instance
        pt14_joint_list.append(new_joint)

    return np.array(pt14_joint_list)


def coco13_2_vmo(coco_kps: Union[list, np.ndarray]) -> np.ndarray:
    """ convert from coco-13 keypoint format to vmo-14 format

    VMO has 14 keypoints. Here based on the revised COCO-13 keypoints,
    the keypoints are remapped with new sequence.
    1). Add one NECK keypoint, which marks the middle point between
        the 2 should keypoints.

    Args:
        coco_kps: array of keypoints in coco-13 format

    Returns:
        array of keypoints in vmo-14 format
    """
    coco_kps = keypoint_array(coco_kps)
    pt14_joint_list = []
    transform = list(
        zip(
            [0, 1, 2, 4, 6, 1, 3, 5, 8, 10, 12, 7, 9, 11],
            [0, 2, 2, 4, 6, 1, 3, 5, 8, 10, 12, 7, 9, 11],
        )
    )
    for prev_joint in coco_kps:
        new_joint = []
        for idx1, idx2 in transform:
            j1 = prev_joint[idx1]
            j2 = prev_joint[idx2]
            new_joint.append(
                ((j1[0] + j2[0]) / 2, (j1[1] + j2[1]) / 2, min(j1[2], j2[2]))
            )
        pt14_joint_list.append(new_joint)

    return np.array(pt14_joint_list)


def coco13_2_torso(coco_kps: Union[list, np.ndarray]) -> np.ndarray:
    """ convert from coco-13 keypoint format to torso-4 format

    Torso-4 has 4 keypoints. Here based on the revised COCO-13 keypoints,
    the keypoints are remapped with new sequence.

    Args:
        coco_kps: array of keypoints in coco-13 format

    Returns:
        array of keypoints in torse-4 format
    """
    coco_kps = keypoint_array(coco_kps)
    pt4_joint_list = []
    transform = list(zip([0, 2, 1, 7], [0, 2, 1, 8]))
    for prev_joint in coco_kps:
        new_joint = []
        for idx1, idx2 in transform:
            j1 = prev_joint[idx1]
            j2 = prev_joint[idx2]
            new_joint.append(
                ((j1[0] + j2[0]) / 2, (j1[1] + j2[1]) / 2, min(j1[2], j2[2]))
            )
        pt4_joint_list.append(new_joint)

    return np.array(pt4_joint_list)


SUPPORTED_FORMATS = ["coco", "coco13", "vmo", "torso"]


CONVERSIONS_Registry = {
    "coco-openpose": coco_2_openpose,
    "coco-vmo": coco_2_vmo,
    "coco13-vmo": coco13_2_vmo,
    "coco13-torso": coco13_2_torso,
}
