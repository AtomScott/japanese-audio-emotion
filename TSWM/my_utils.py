import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def imshow(img, bboxes=None):
    fig,ax = plt.subplots(1)
    ax.imshow(img.transpose(1,2,0).astype(np.int))

    for bbox in bboxes:
        y1,x1,y2,x2 = bbox
        h = y2 - y1
        w = x2 - x1
        rect = patches.Rectangle((x1,y1),w,h,linewidth=1,edgecolor='g',facecolor='none')
        ax.add_patch(rect)
    ax.axis('off')
    plt.show()

def get_iou(a, b, epsilon=1e-5):
    """ Given two boxes `a` and `b` defined as a list of four numbers:
            [y1,x1,y2,x2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.

    Args:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (float) The Intersect of Union score.
    """
    # COORDINATES OF THE INTERSECTION BOX
    y1 = max(a[0], b[0])
    x1 = max(a[1], b[1])
    y2 = min(a[2], b[2])
    x2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap
    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon)
    return iou

def score_face_detection(bboxes, labels):
    score = 0
    for bbox in bboxes:
        max_iou = 0.
        for label in labels:
            iou = get_iou(bbox, label)
            max_iou = iou if iou > max_iou else max_iou
        score += 1 if max_iou >= 0.5 else 0
    return score
