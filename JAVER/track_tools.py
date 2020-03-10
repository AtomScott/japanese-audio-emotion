import more_itertools as mit

import torch
import imageio

from tqdm import tqdm

from facenet_pytorch import MTCNN, InceptionResnetV1
from facenet_pytorch.models import utils as facenet_utils
from facenet_pytorch.models.mtcnn import prewhiten


class FaceTracker():
    def __init__(self, image_size, video_path, batch_size, step):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.detector = MTCNN(image_size=image_size,
                              keep_all=True, device=device)
        self.video = imageio.get_reader(video_path,  'ffmpeg')
        self.batch_size = batch_size
        self.step = step

    def track(self):
        def _track(step):
            images = []
            for i in range(batch_size * step):
                image = video.get_next_data()
                if i % step == 0:
                    images.append(image)
            boxes, probs = self.detect(images)
            return boxes, probs

        video = self.video
        batch_size = self.batch_size
        step = self.step

        video.set_image_index(0)
        n_frames = video.count_frames()

        i = 0
        while i < n_frames:
            # Face Detection
            boxes, probs = _track(step=step)
            first_true = mit.first_true(probs, default=-1, pred=lambda x: x is not None)
            first_true = probs.index(first_true) if first_true != -1 else -1
            i += step*batch_size

            # Face is detected
            if first_true > -1:  
                if step == 1:
                    pass
                elif step == self.step:
                    step = 1
                    i = first_true
                    self.video.set_image_index(i)

            # Final frame is not a face
            if first_true == batch_size:  
                step = self.step

    def detect(self, images, threshold=0.98):
        bboxes, probs = self.detector.detect(images)
        # TODO bboxes 
        thresh_bboxes, thresh_probs = [], []
        for  _bboxes, _probs in zip(bboxes, probs):
            if _bboxes is None:
                continue
            _thresh_bboxes, _thresh_probs = [], []
            for bbox, prob in zip(_bboxes, _probs):
                if prob > threshold:
                    _thresh_bboxes.append(bbox)
                    _thresh_probs.append(prob)
                thresh_bboxes.append(_thresh_bboxes)
                thresh_probs.append(_thresh_probs)
        return thresh_bboxes, thresh_probs


        bboxes = [bbox for _bboxes, _probs in zip(bboxes, probs) if prob > threshold]
        probs = [prob for bboxes, probs in zip(bboxes, probs) if prob > threshold]
        return bboxes, probs

    def get_roi(self, image, bbox):
        x1,y1,x2,y2 = bbox
        bbox = (
            min(x1, x2),# left
            min(y1, y2),# lower
            max(x1, x2),# right
            max(y1, y2) # upper
            )

        image_size = self.image_size
        roi = image.crop(bbox).resize((image_size, image_size))
        return roi

