from enum import Enum


class JointFormat(Enum):
    coco = 0
    openpose = 1
    openpose20 = 2
    openpose22 = 3
    vmo = 4
    vmo16 = 5
    vmo18 = 6
    torso = 7


class TorsoPart(Enum):
    Head = 0
    RShoulder = 1
    LShoulder = 2
    Hip = 3


class VmoPart(Enum):
    Head = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13


class CocoPart(Enum):
    Nose = 0
    LEye = 1
    REye = 2
    LEar = 3
    REar = 4
    LShoulder = 5
    RShoulder = 6
    LElbow = 7
    RElbow = 8
    LWrist = 9
    RWrist = 10
    LHip = 11
    RHip = 12
    LKnee = 13
    RKnee = 14
    LAnkle = 15
    RAnkle = 16


class Coco13Part(Enum):
    Head = 0
    LShoulder = 1
    RShoulder = 2
    LElbow = 3
    RElbow = 4
    LWrist = 5
    RWrist = 6
    LHip = 7
    RHip = 8
    LKnee = 9
    RKnee = 10
    LAnkle = 11
    RAnkle = 12


class OpenposePart(Enum):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    RHip = 8
    RKnee = 9
    RAnkle = 10
    LHip = 11
    LKnee = 12
    LAnkle = 13
    REye = 14
    LEye = 15
    REar = 16
    LEar = 17
    Background = 18


coco13_flip_list = [
    Coco13Part.Head,
    Coco13Part.RShoulder,
    Coco13Part.LShoulder,
    Coco13Part.RElbow,
    Coco13Part.LElbow,
    Coco13Part.RWrist,
    Coco13Part.LWrist,
    Coco13Part.RHip,
    Coco13Part.LHip,
    Coco13Part.RKnee,
    Coco13Part.LKnee,
    Coco13Part.RAnkle,
    Coco13Part.LAnkle,
]


nr_torso_keypoints = 4  # 4 joints
torso_skeleton = [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3), (1, 3)]
torso_flip_list = [
    TorsoPart.Head,
    TorsoPart.LShoulder,
    TorsoPart.RShoulder,
    TorsoPart.Hip,
]
nr_torso_links = 5
torsopose_vecs = list(zip([0, 0, 1, 1, 2], [1, 2, 2, 3, 3]))

nr_vmo_keypoints = 14  # 14 joints
vmo_skeleton = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (1, 5),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
]
vmo_flip_list = [
    VmoPart.Head,
    VmoPart.Neck,
    VmoPart.LShoulder,
    VmoPart.LElbow,
    VmoPart.LWrist,
    VmoPart.RShoulder,
    VmoPart.RElbow,
    VmoPart.RWrist,
    VmoPart.LHip,
    VmoPart.LKnee,
    VmoPart.LAnkle,
    VmoPart.RHip,
    VmoPart.RKnee,
    VmoPart.RAnkle,
]
# fmt: off
nr_vmo_links = 14
vmopose_vecs = list(
    zip(
        [1,1,8, 9, 1,11,12,1,2,3,1,5,6, 8],
        [0,8,9,10,11,12,13,2,3,4,5,6,7,11],
    )
)

nr_vmo16_links = 16
vmopose16_vecs = list(
    zip(
        [1,1,8, 9, 1,11,12,1,2,3,1,5,6, 8,2,5],
        [0,8,9,10,11,12,13,2,3,4,5,6,7,11,0,0],
    )
)

nr_vmo18_links = 18
vmopose18_vecs = list(
    zip(
        [1,1,8, 9, 1,11,12,1,2,3,1,5,6, 8,2,5,2, 5],
        [0,8,9,10,11,12,13,2,3,4,5,6,7,11,0,0,8,11],
    )
)
# fmt: on

nr_coco_keypoints = 17
coco_skeleton = [
    (15, 13),
    (13, 11),
    (16, 14),
    (14, 12),
    (11, 12),
    (5, 11),
    (6, 12),
    (5, 6),
    (5, 7),
    (6, 8),
    (7, 9),
    (8, 10),
    (1, 2),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (3, 5),
    (4, 6),
]
cocoorg_flip_list = [
    CocoPart.Nose,
    CocoPart.REye,
    CocoPart.LEye,
    CocoPart.REar,
    CocoPart.LEar,
    CocoPart.RShoulder,
    CocoPart.LShoulder,
    CocoPart.RElbow,
    CocoPart.LElbow,
    CocoPart.RWrist,
    CocoPart.LWrist,
    CocoPart.RHip,
    CocoPart.LHip,
    CocoPart.RKnee,
    CocoPart.LKnee,
    CocoPart.RAnkle,
    CocoPart.LAnkle,
]

nr_openpose_keypoints = 18  # 18 joints
openpose_skeleton = [
    (1, 2),
    (1, 5),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (1, 8),
    (8, 9),
    (9, 10),
    (1, 11),
    (11, 12),
    (12, 13),
    (1, 0),
    (0, 14),
    (14, 16),
    (0, 15),
    (15, 17),
]  # = 17
openpose_flip_list = [
    OpenposePart.Nose,
    OpenposePart.Neck,
    OpenposePart.LShoulder,
    OpenposePart.LElbow,
    OpenposePart.LWrist,
    OpenposePart.RShoulder,
    OpenposePart.RElbow,
    OpenposePart.RWrist,
    OpenposePart.LHip,
    OpenposePart.LKnee,
    OpenposePart.LAnkle,
    OpenposePart.RHip,
    OpenposePart.RKnee,
    OpenposePart.RAnkle,
    OpenposePart.LEye,
    OpenposePart.REye,
    OpenposePart.LEar,
    OpenposePart.REar,
]  # , OpenposePart.Background]

# fmt: off
nr_openpose_links = 19
openpose_vecs = list(
    zip(
        [1,8, 9, 1,11,12,1,2,3,1,5,6, 5,1, 0, 0,14, 2,15],
        [8,9,10,11,12,13,2,3,4,5,6,7,17,0,14,15,16,16,17],
    )
)

nr_openpose20_links = 20
openpose20_vecs = list(
    zip(
        [1,8, 9, 1,11,12, 8,1,2,3,1,5,6, 5,1, 0, 0,14, 2,15],
        [8,9,10,11,12,13,11,2,3,4,5,6,7,17,0,14,15,16,16,17],
    )
)

nr_openpose22_links = 22
openpose22_vecs = list(
    zip(
        [1,8, 9, 1,11,12, 8,2, 5,1,2,3,1,5,6, 5,1, 0, 0,14, 2,15],
        [8,9,10,11,12,13,11,8,11,2,3,4,5,6,7,17,0,14,15,16,16,17],
    )
)
# fmt: on
