import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import chainercv
import face_recognition

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
        if bbox == []:
            continue
        max_iou = 0.
        for label in labels:
            iou = get_iou(bbox, label)
            max_iou = iou if iou > max_iou else max_iou
        score += 1 if max_iou >= 0.5 else 0
    return score

def crop_bbox(img, bbox):
    """
    Crops an image to a given bbox.

    Args:
        img: Image to crop
        bbox: [y1, x1, y2, x2] shape

    Returns:
        type: Cropped image

    Raises:
        Exception: description

    """
    bbox = bbox.flatten().astype('int32')
    cropped_img = img[:, bbox[0]:bbox[2], bbox[1]:bbox[3]]
    return cropped_img


def unravel_list_dict(ld):
    """
    Unravels a list of ditctionaries to a 1-d numpy array.

    Args:
        A list of dictionaries.

    Returns:
        type: 1-d array

    Raises:
        Exception: description

    """
    l = []
    for d in ld:
        for key, value in d.items():
            l += value

    arr = np.asarray(l).ravel()
    return arr


def vis_points(img, points, ax=None, params={}):
    """Visualize points in an image.

    Args:
        img: An image (3, W, H)

        points: A list of Points to Visualize (M, N, 2)

        ax: Matplotlib Axes

        params: Dictionary of params for matplotlib
    Returns:

    """
    cm = plt.get_cmap('gist_rainbow')

    with plt.rc_context(params):

        # Returns newly instantiated matplotlib.axes.Axes object if ax is None
        ax = chainercv.visualizations.vis_image(img, ax=ax)

        _, H, W = img.shape
        n_inst = len(points)

        for i in range(n_inst):
            pnts = points[i]
            n_point = len(pnts)

            colors = [cm(k / n_point) for k in range(n_point)]

            for k in range(n_point):
                ax.scatter(pnts[k][0], pnts[k][1], c=[colors[k]], s=100)

        ax.set_xlim(left=0, right=W)
        ax.set_ylim(bottom=H - 1, top=0)
    return ax

def draw_facial_landmark(inpath, outpath):
    """
    Draw facial landmarks onto images.

    Args:
        variable (type): description

    Returns:
        type: description

    Raises:
        Exception: description

    """
    # read image
    img = chainercv.utils.read_image(inpath)

    # Do facial landmark detection
    face_landmarks_list = face_recognition.face_landmarks(img.transpose(1,2,0).astype('uint8'))
    points = unravel_list_dict(face_landmarks_list)
    points = points.reshape(len(points)//2, 2)

    # Draw the image
    fig = plt.figure()

    ax = plt.axes([0,0,1,1], frameon=False)
    fig.add_axes(ax)

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.autoscale(tight=True)


    params = {'scatter.marker': '.',
             'savefig.pad_inches': 0}
    vis_points(img, [points], ax=ax, params=params)

    plt.axis('off')
    plt.savefig(outpath)
    plt.show()
