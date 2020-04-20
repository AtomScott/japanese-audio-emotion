import numpy as np
from facenet_pytorch import MTCNN
from JAVER.utils.logger import create_logger

def detect_faces(images, threshold=0.5):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)

    bboxes_found, probs_found = mtcnn.detect(images)

    bboxes_above_thresh = []
    for bboxes, probs in zip(bboxes_found, probs_found):
        if bboxes is None:
            bboxes_above_thresh.append(np.asarray([]))
        else:
            _bboxes = []
            for idx, (bbox, prob) in enumerate(zip(bboxes, probs)):
                if prob >= threshold:
                    x1, y1, x2, y2 = list(map(int, bbox))

                    _bboxes.append([x1, y1, x2, y2])
                    logger.debug(f'{idx} {x1, y1}, {x2, y2}, {prob:.4f}')

            bboxes = np.asarray(_bboxes)
            bboxes_above_thresh.append(bboxes)

    return np.asarray(bboxes_above_thresh)
