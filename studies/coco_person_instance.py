from absl import app
import sys
from pathlib import Path

proj_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(proj_root))

import cv2
import json

from uniview.IM2D.bbox import UnviewBBox
from uniview.IM2D import coco_person as cp


def main(_argv):

    # env
    home = Path("~").expanduser()

    # load test data
    with open("local_data/mini_coco.json") as f:
        mini_coco = json.load(f)
    print(type(mini_coco))

    # try demo functions
    """ annotation keys:
    ['segmentation', 'num_keypoints', 'area', 'iscrowd', 'keypoints',
     'image_id', 'bbox', 'category_id', 'id'] """

    unibox = UnviewBBox()

    for imf, annotations in mini_coco.items():
        print(f"---------- {imf} ------------")
        im = cv2.imread(str(home / imf))
        print(im.shape)
        print(type(im.shape))
        bboxes, keypoints = [], []
        for inst in annotations:
            bboxes.append(inst["bbox"])
            keypoints.append(inst["keypoints"])
        bboxes = unibox.get_albu_bbox(bboxes, "coco", im.shape)

        im = cp.draw_boxes(im, bboxes)
        im = cp.vis_keypoints(im, keypoints)
        cv2.imshow("example", im)
        cv2.waitKey(0)


if __name__ == "__main__":
    """example:
        python studies/coco_person_instance.py
    """
    try:
        app.run(main)
    except SystemExit:
        pass
