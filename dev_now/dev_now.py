from absl import app
import sys
from pathlib import Path

proj_root = Path(__file__).parent.parent.absolute()
sys.path.append(str(proj_root))

from uniview.IM2D.bbox import UniviewBBox


def main(_argv):

    # mock data
    # bboxes = [[0.1, 0.8, 0.4, 0.5], [0.9, 0.8, 0.4, 0.5]]
    # boxformat = 'yolo'

    # bboxes = [[ -10., 55., 40., 144.], [70., 55., 150., 60.]]
    # boxformat = 'coco'

    bboxes = [-10.0, 55.0, 40.0, 144.0]
    boxformat = "coco"

    imshape = (100, 200)  # (height, width, detph)

    unibox = UniviewBBox(need_trim=True)
    nbb = unibox.get_albu_bbox(bboxes, boxformat, imshape)
    print(nbb)


if __name__ == "__main__":
    """example:
        python studies/dev_now.py
    """
    try:
        app.run(main)
    except SystemExit:
        pass
