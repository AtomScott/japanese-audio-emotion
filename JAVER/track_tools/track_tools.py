import itertools as it
from collections import defaultdict

import PIL
import cv2
import mmcv
import numpy as np
import torch
from PIL import Image
from cvt.models import SubspaceMethod
from facenet_pytorch import MTCNN, InceptionResnetV1
from facenet_pytorch.models import utils as facenet_utils
from facenet_pytorch.models.mtcnn import prewhiten
from scipy.spatial.distance import mahalanobis
from scipy.optimize import linear_sum_assignment
from scipy.interpolate import interp1d
from .logger import create_logger
from moviepy.editor import VideoFileClip

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=True, device=device)
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
        self.bbox = list(map(int, bbox))

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


class Track:
    def __init__(self, face):
        # Kalman stuff
        x1, y1, x2, y2 = face.bbox
        x = np.mean((x1, x2))
        y = np.mean((y1, y2))
        s = np.linalg.norm((x1 - x2, y1 - y2))

        self.dt = dt = 0.1

        self.state_x = np.array([x, y, s, 0, 0, 0])
        self.state_prev_x = self.state_x

        self.state_cov = P = np.diag(np.ones(self.state_x.shape))

        self.H = np.asarray([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ])

        # confidence probably needs tuning
        conf = 1
        self.R = np.diag(np.ones(len(self.state_x))) * conf

        # Deep sort stuff
        self.gallery = []
        self.none_count = 0
        pass

    def predict_state(self):
        x_now = self.state_x
        P_now = self.state_cov
        H = self.H

        x_pred = H @ x_now
        P_pred = H @ P_now @ H.T

        return x_pred, P_pred

    def update(self, face):
        if face is None:
            self.none_count += 1
        else:
            self.update_gallery(face)
            self.update_state(face)
            self.none_count = 0
        return

    def update_state(self, face):
        z = self.format_measurement(face)

        x_now = self.state_x
        P_now = self.state_cov

        H = self.H
        R = self.R

        K = P_now @ H.T @ np.linalg.inv(H @ P_now @ H.T + R)

        x_next = x_now + K @ (z - H @ x_now)
        P_next = P_now - K @ H @ P_now

        self.state_prev_x = self.state_x
        self.state_x = x_next
        self.state_cov = P_next / np.linalg.norm(P_next)

        return

    def format_measurement(self, face):
        x1, y1, x2, y2 = face.bbox
        _, _, _, x_prev, y_prev, s_prev = self.state_prev_x

        x = np.mean((x1, x2))
        y = np.mean((y1, y2))
        s = np.linalg.norm((x1 - x2, y1 - y2))
        xv = x - x_prev
        yv = y - y_prev
        sv = s - s_prev
        z = np.array([x, y, s, xv, yv, sv])
        return z

    def update_gallery(self, face):
        self.gallery.append(face)
        pass

    def get_bboxes(self):
        """
        x1, y1, x2, y2 = get_bboxes(track)
        """

        start_t = self.gallery[0].idx
        end_t = self.gallery[-1].idx

        xs = np.zeros((end_t + 1 - start_t, 4))

        idxs_old = []
        bboxes_old = []
        for face in self.gallery:
            idxs_old.append(face.idx)
            bboxes_old.append(face.bbox)

        idxs_new = np.arange(start_t, end_t)
        bboxes_new = []

        ys = np.asarray(bboxes_old)
        for i in range(4):
            f = interp1d(np.asarray(idxs_old), ys[:, i])
            bboxes_new.append(f(idxs_new))

        return np.asarray(bboxes_new)



