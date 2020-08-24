from absl import app
import sys
from pathlib import Path

proj_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(proj_root))

import cv2
import json

# from tools.visualization.im2d_viz_utils import draw_boxes, vis_keypoints


# def demo_example(im, bboxes, cats=None, kps=[], resize=1.0):
#     bboxes = [nbbx_2_ncornerbox(b) for b in bboxes]
#     im = draw_boxes(im, bboxes, cats)
#     im = vis_keypoints(im, kps)
#     if resize != 1.0:
#         im = cv2.resize(im, (0, 0), fx=resize, fy=resize)
#     return im


def main(_argv):

    # env
    home = Path("~").expanduser()

    # load test data
    with open("local_data/mini_coco.json") as f:
        mini_coco = json.load(f)
    print(type(mini_coco))

    # try demo functions
    for imf, annotations in mini_coco.items():
        im = cv2.imread(str(home / imf))
        print(im.shape)
        print(type(im.shape))


if __name__ == "__main__":
    """example:
        python studies/coco_person_instance.py
    """
    try:
        app.run(main)
    except SystemExit:
        pass
