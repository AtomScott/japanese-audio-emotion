import more_itertools as mit

import torch
import imageio

from tqdm import tqdm

from facenet_pytorch import MTCNN, InceptionResnetV1
from facenet_pytorch.models import utils as facenet_utils
from facenet_pytorch.models.mtcnn import prewhiten


class FaceTracker():
    """
    Tracks faces of a single person for a given range in a video.
    """

    def __init__(self, image_size: int, batch_size: int, step: int):

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.detector = MTCNN(image_size=image_size,
                              keep_all=True, device=device)
        self.batch_size = batch_size
        self.step = step

    def track(self, video_path: list, reference_images: list):
        """Tracks images of a single person in a given video path

        Parameters
        ----------
        out_path : list
            The output paths for each video
        video_path : list
            Paths to video
        reference_images : list
            Paths to images to use as reference for face recognition

        Returns
        -------
        [type]
            A list of bboxes corresponding to each image frame?
        """

        """
        1. Get bboxes and rois for every frame
        """
        bboxes, rois = detect()

        """
        2. Get embeddings for every roi
        """
        embeddings = embed(rois)

        """
        3. Associate faces and create trajectories
        """
        return associate(embeddings)
        
        def _track():
            """Helper function that returns bboxes over a batch of images
            """
            images = []
            for i in range(batch_size * step):
                image = video.get_next_data()
                if i % step == 0:
                    images.append(image)
            boxes, probs = self.detect(images)
            rois = [self.get_roi(box, image) for box in boxes]
            return boxes, rois

        batch_size = self.batch_size
        step = self.step

        video = imageio.get_reader(video_path,  'ffmpeg')
        n_frames = video.count_frames()
        
        # return trajectories
        trajectories = []

        i = 0 
        while i < n_frames:
            # Face Detection
            boxes, rois = _track(step=step)
            i += step*batch_size

            # Find first frame to find face
            first_true = mit.first_true(
                rois, default=-1, pred=lambda x: x is not None)
            first_true_idx = rois.index(first_true) if first_true != -1 else -1
            
            # Face is detected during rapid search
            if first_true >= 0 and step == self.step:

                # Start detection at index first_true with step size = 1
                step = 1

                # Role back to first detection
                i = first_true_idx
                self.video.set_image_index(i)



            # Final frame is not a face
            elif first_true == -1:

                #  Increase step size and do rapid search
                step = self.step

            # Do data association if frames are continuous
            elif step == 1:
                
                bbox = self.associate(boxes, rois)
                bboxes.append(bbox)

        return trajectories

    def detect(self, images, threshold=0.98):
        bboxes, probs = self.detector.detect(images)
        # TODO bboxes
        thresh_bboxes, thresh_probs = [], []
        for _bboxes, _probs in zip(bboxes, probs):
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

        bboxes = [bbox for _bboxes, _probs in zip(
            bboxes, probs) if prob > threshold]
        probs = [prob for bboxes, probs in zip(
            bboxes, probs) if prob > threshold]
        return bboxes, probs

    def get_roi(self, image, bbox):
        x1, y1, x2, y2 = bbox
        bbox = (
            min(x1, x2),  # left
            min(y1, y2),  # lower
            max(x1, x2),  # right
            max(y1, y2)  # upper
        )

        image_size = self.image_size
        roi = image.crop(bbox).resize((image_size, image_size))
        return roi

class Trajectory():
    def __init__(self):
        self.state
        self.gallery
        pass

    def update_state(self):
        pass

    def update_gallery(self):
        pass
