import more_itertools as mit

import torch
import imageio
import PIL
import numpy as np

from tqdm import tqdm
from PIL import Image

from cvt.models import SubspaceMethod

from facenet_pytorch import MTCNN, InceptionResnetV1
from facenet_pytorch.models import utils as facenet_utils
from facenet_pytorch.models.mtcnn import prewhiten

from .logger import create_logger

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

logger = create_logger(level='DEBUG')


def format_input(X, y):
    X = [X[np.where(y == t)] for t in np.unique(y)]
    return X, np.unique(y)


class Face:
    def __init__(self, idx, img, bbox, is_face=False):

        if type(img) is not PIL.JpegImagePlugin.JpegImageFile:
            img = np.asarray(img)
            img = Image.fromarray(img)

        self.img = img
        if is_face:
            bbox = (0, 0, img.width, img.height)

        self.idx = idx
        self.bbox = bbox

        if self.is_valid():
            self.face_img = self.get_roi()
            self.embedding = self.get_embedding()

        else:
            self.face = None
            self.embedding = None

    def is_valid(self):
        try:
            x1, y1, x2, y2 = self.bbox
            return True
        except:
            return False


    def get_roi(self):
        return self.img.crop(self.bbox)

    def get_embedding(self):

        tensor = facenet_utils.detect_face.extract_face(
            self.img, box=self.bbox, image_size=160)

        aligned = torch.stack([prewhiten(tensor)]).to(device)

        embedding = resnet(aligned).detach().cpu().detach().numpy()[0]
        return embedding


class Trajectory:
    def __init__(self):
        self.state
        self.gallery
        pass

    def update_state(self):
        pass

    def update_gallery(self):
        pass


class FrameHandler:
    def __init__(self, step, batch_size, video_path):
        self.idx = 0
        self.step = step
        self.batch_size = batch_size

        reader = imageio.get_reader(video_path,  'ffmpeg')
        frames = []
        for i in reader:
            frames.append(reader.get_next_data())
        self.frames = frames

    def __len__(self):
        return len(self.frames)

    def next(self):
        if self.has_next():
            start_idx = self.idx
            end_idx = start_idx + (self.step * self.batch_size)
            self.idx = end_idx

            # logger.debug(f'idx: {start_idx}~{end_idx}')

            if end_idx <= self.__len__():
                return self.frames[start_idx:end_idx:self.step]
            else:
                return self.frames[start_idx:-1:self.step]
        else:
            return None

    def has_next(self):
        return self.idx <= self.__len__()


class FaceTracker:
    """
    Tracks faces of a single person for a given range in a video.
    """

    def __init__(self, image_size: int, ref_paths, batch_size: int, step: int):
        self.batch_size = batch_size
        self.step = step

        ref_faces = [Face(None, Image.open(path), None, is_face=True)
                     for path in ref_paths]
        x_train = np.asarray([face.embedding for face in ref_faces])
        y_train = 1

        SM = SubspaceMethod(n_subdims=5)
        SM.fit([x_train], [y_train])
        self.clf = SM

    def track(self, video_path: list):
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
        faces = self.detect(reference_images, video_path)

        # group bboxes by frame

        """
        2. Associate faces and create trajectories
        """
        return self.associate(faces)

    def detect(self, video_path, step_large=100, step_small=10):
        frame_iter = FrameHandler(step_large, self.batch_size, video_path)
        found_faces = {i:[] for i in range(0, len(frame_iter), step_small)}

        while frame_iter.has_next():

            head_idx = frame_iter.idx
            frames = frame_iter.next()

            faces = []
            for i, (frame, bboxes) in enumerate(zip(frames, self.detect_faces(frames))):
                idx=frame_iter.step * i+head_idx
                if bboxes is None:
                    logger.debug(f'Frame: {idx}, n_faces: {0}')
                    continue
                for bbox in bboxes:
                    face = Face(idx=idx, img=frame, bbox=bbox)
                    faces.append(face)
                logger.debug(f'Frame: {idx}, n_faces: {len(bboxes)}')

            # target_faces = list(filter(self.is_reference_face, faces))
            # logger.debug(f'Faces/Frames = {len(faces)}/{len(frames)}')
            # for i in range(len(frames)):
            #     idx = i + head_idx
            #     logger.debug(f'Frame: {idx}, n_faces: {faces[i]}')

            if len(faces) == 0:
                frame_iter.step = step_large

            elif frame_iter.step == step_large:
                rollback_id = faces[0].idx

                if rollback_id > -1:
                    idx = rollback_id - frame_iter.step if rollback_id != 0 else 0
                    logger.info(f'Rollback to {idx}')
                    frame_iter.idx = idx
                    frame_iter.step = step_small

            elif frame_iter.step == step_small:
                rollforward_id = faces[-1].idx \
                    if faces[-1].idx != frame_iter.idx else -1

                if rollforward_id > -1:
                    idx = rollforward_id + frame_iter.step
                    logger.info(f'Rollforward to {idx}')
                    frame_iter.idx = idx
                    frame_iter.step = step_large

                for face in faces:
                    # print(face.bbox in [ff.bbox for ff in found_faces[face.idx]])
                    if found_faces[face.idx] == []:
                        pass
                    elif face.bbox.astype(int).tolist() in  [ff.bbox.astype(int).tolist() for ff in found_faces[face.idx]]:
                        continue
                    found_faces[face.idx].append(face)

        return found_faces

    def detect_faces(self, images, threshold=0.95):
        bboxes, probs = mtcnn.detect(images)
        return bboxes

        thresh_bboxes = []
        for _bboxes, _probs in zip(bboxes, probs):
            if _bboxes is None:
                continue
            _thresh_bboxes = []

            for bbox, prob in zip(_bboxes, _probs):
                if prob > threshold:
                    _thresh_bboxes.append((tuple(bbox)))

                thresh_bboxes.append(_thresh_bboxes)
        return thresh_bboxes

    def is_reference_face(self, face):
        if face.is_valid():
            probs = self.clf.predict_proba(face.embedding).squeeze()
            return probs > 0.5
        else:
            return False

    def associate(self, face_seq):
        for faces in face_seq:
            C = cost_matrix(live_trajs, faces)
            G = gate_matrix(live_trajs, faces)


        return
