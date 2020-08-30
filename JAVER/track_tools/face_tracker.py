"""

"""

import PIL
from PIL import Image
import numpy as np
import torch
import os
from cvt.models import SubspaceMethod
from JAVER.track_tools.track_tools import Face, Track
from JAVER.track_tools.frame_handler import FrameHandler
from collections import defaultdict
from moviepy.editor import VideoFileClip
from scipy.optimize import linear_sum_assignment

from facenet_pytorch import MTCNN, InceptionResnetV1
from facenet_pytorch.models import utils as facenet_utils
from facenet_pytorch.models.mtcnn import prewhiten
from JAVER.utils.logger import create_logger
from JAVER.track_tools.metrics import association_cost_matrix, gate_matrix

logger = create_logger(level='DEBUG')


class FaceTracker:
    """
    Track faces of a single person for a given video.

        Example
       --------
        >>> face_tracker = FaceTracker(160, "Elon_img.png", 1, 3, 1)
        >>> face_tracker.track("Elon_vid.mp4")

    """
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)

    def __init__(self, image_size: int, ref_paths: list, batch_size: int, step_large: int, step_small: int):
        self.batch_size = batch_size
        self.step_large = step_large
        self.step_small = step_small

        ref_faces = [Face(None, Image.open(path), None, is_face=True)
                     for path in ref_paths]
        x_train = np.asarray([face.embedding for face in ref_faces])
        y_train = 1

        SM = SubspaceMethod(n_subdims=5)
        SM.fit([x_train], [y_train])
        self.clf = SM

    def track(self, video_paths: list, out_dir: str):
        """Tracks the face of a single person in a video.

        Parameters
        ----------
        video_paths : list
            Paths to video
        out_dir : list
            The output path for each video

        Returns
        -------
        [type]
            A list of bboxes corresponding to each image frame?

        """

        for video_path in video_paths:
            ################
            # * Detect faces
            ################

            faces_dict = self.detect(video_path)

            ###################
            # * Associate faces
            ###################

            tracks = self.associate(faces_dict)
            logger.info(f"{len(tracks)} found.")

            ###############
            # * Save tracks
            ###############

            for i, track in enumerate(tracks):
                clip = VideoFileClip(video_path)
                x1s, y1s, x2s, y2s = track.get_bboxes()

                def fx(t):
                    if t < len(x1s):
                        return x1s[t], x2s[t]
                    else:
                        return x1s[len(x1s)], x2s[len(x1s)]

                def fy(t):
                    if t < len(y1s):
                        return y1s[t], y2s[t]
                    else:
                        return y1s[len(y1s)], y2s[len(y1s)]

                if len(x1s) == 0: logger.warning('x1s is empty')
                if len(x2s) == 0: logger.warning('x2s is empty')
                if len(y1s) == 0: logger.warning('y1s is empty')
                if len(y2s) == 0: logger.warning('y2s is empty')

                start_t = track.get_track_start() / 25
                end_t = track.get_track_end() / 25
                logger.debug(f"start_t: {start_t}, end_t: {end_t}")
                subclip = clip.subclip(start_t, end_t)
                subclip = subclip.fx(moving_crop, fx=fx, fy=fy)

                subclip.write_videofile(os.path.join(out_dir, f"{i}.mp4"), fps=25)
            return None

    def detect(self, video_path: str) -> dict:
        """Finds the faces in a video. Frames are skipped according to self.step_small
        and self.step_large.
        
        Parameters
        ----------
        video_path : str
            A path to a video file.
        
        Returns
        -------
        faces_dict : dict
            A dictionary containing frame index (key) and list of Face objects
            (value).
        """
        step_large = self.step_large
        step_small = self.step_small
        assert step_small < step_large, "step_small must be smaller than step_large"

        batch_size = self.batch_size

        self.frame_handler = frame_handler = FrameHandler(
            step_large, batch_size, video_path)
        faces_dict = defaultdict(lambda: [])

        # {i: [] for i in range(0, len(frame_handler), step_small)}

        while frame_handler.has_next():

            # Get next batch
            head_idx = frame_handler.idx
            frames = frame_handler.next()
            tail_idx = frame_handler.idx

            # Detect all bboxes in batch
            bboxes_found_in_batch = self.detect_faces(frames)

            if frame_handler.step == step_large and bboxes_found_in_batch.size != 0:

                # find first face in batch
                first_face_idx = next(
                    filter(lambda x: x[1].size, enumerate(bboxes_found_in_batch)))[0]
                rollback_idx = head_idx + frame_handler.step * first_face_idx

                # roll back to rollback_idx
                logger.info(f'Rollback to {rollback_idx}')
                frame_handler.idx = rollback_idx

                # Change step size to small
                frame_handler.step = step_small

            elif frame_handler.step == step_small:

                # Init all faces from bboxes and frames
                for i, bboxes in enumerate(bboxes_found_in_batch):
                    frame = frames[i]
                    idx = head_idx + (frame_handler.step * i)

                    for bbox in bboxes:
                        face = Face(idx=idx, img=frame, bbox=bbox)
                        faces_dict[idx].append(face)

                    n_faces = len(bboxes)
                    logger.debug(f'Frame: {idx}, n_faces: {n_faces}')
                    rollforward_idx = idx + frame_handler.step
                    if n_faces == 0:
                        break

                if rollforward_idx != tail_idx:
                    logger.info(
                        f'Rollforward to {rollforward_idx} (Current tail @ {tail_idx})')
                    frame_handler.idx = rollforward_idx
                    frame_handler.step = step_large

        logger.info("Finished Tracking")
        return faces_dict

    def detect_faces(self, images: PIL.Image, threshold: float = 0.85) -> np.ndarray:
        """Detect faces using an MTCNN detector
        
        Parameters
        ----------
        images : PIL.Image
            [description]
        threshold : float, optional
            [description], by default 0.85
        
        Returns
        -------
        np.ndarray
            A 2d list containing bounding boxes for each frame.
        """
        bboxes_found, probs_found = FaceTracker.mtcnn.detect(images)

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

    def is_reference_face(self, face):
        if face.is_valid():
            probs = self.clf.predict_proba(face.embedding).squeeze()
            return probs > 0.5
        else:
            return False

    def associate(self, faces_dict):

        salt = 1 / 10 ** 9  # avoids zero division
        pepper = 10 ** 6  # avoids infinity

        tracks_alive = []
        tracks_dead = []

        prev_frame_idx = 0

        ##################
        # * Set parameters
        ##################

        lam = 0.1
        thresh_1, thresh_2 = 0.6, 0.3
        kill_thresh = 2
        frame_idx_thresh = 20  # kill after 20 frame gap

        for frame_idx, faces in faces_dict.items():

            ################
            # * Reset tracks
            ################
            # - Tracks are reset if large gap between previous frame
            if frame_idx - prev_frame_idx > frame_idx_thresh:
                for track in tracks_alive:
                    tracks_dead.append(track)
                tracks_alive = []

            ###########################
            # * Calculate cost matrices
            ###########################
            C = association_cost_matrix(faces, tracks_alive, lam)
            G = gate_matrix(faces, tracks_alive, thresh_1, thresh_2) + salt
            gated_cost_matrix = C / G
            assert C.shape == G.shape, f'{C.shape}, {G.shape}'

            face_idxs, track_idxs = linear_sum_assignment(gated_cost_matrix)
            logger.debug(f"G/C shape: {gated_cost_matrix.shape}, n_tracks: {len(tracks_alive)}")

            ##########################
            # * Assign faces to tracks
            ##########################
            # - faces[face_idx] is assigned to tracks_alive[track_idx]
            # - Remember faces that have not been assigned
            # - Remember tracks that are to be killed
            faces_unassigned = list(face_idxs) if face_idxs != [] else list(range(len(faces)))
            tracks_to_kill = []
            for face_idx, track_idx in zip(face_idxs, track_idxs):
                if gated_cost_matrix[face_idx, track_idx] < pepper:
                    logger.debug(
                        f'Frame: {frame_idx}. Add face {face_idx} to track {track_idx}')
                    tracks_alive[track_idx].update(faces[face_idx])
                    faces_unassigned.remove(face_idx)
                else:
                    tracks_alive[track_idx].update(None)
                    logger.debug(
                        f'Frame: {frame_idx}. Track {track_idx} was not associated to any face. none_count={tracks_alive[track_idx].none_count}')

            for track_idx, track in enumerate(tracks_alive):
                if track_idx not in track_idxs:
                    tracks_alive[track_idx].update(None)
                    logger.debug(
                        f'Frame: {frame_idx}. Track {track_idx} was not associated to any face. none_count={tracks_alive[track_idx].none_count}')
                if track.none_count >= kill_thresh:
                    tracks_dead.append(track)
                    tracks_to_kill.append(track_idx)

            ####################################
            # * Kill tracks/Generate new tracks
            ####################################
            # - Generate new tracks for new faces
            # - Terminate track with continuous null updates
            for track_idx in sorted(tracks_to_kill, reverse=True):
                tracks_dead.append(tracks_alive[track_idx])
                del tracks_alive[track_idx]
                logger.debug(f'Frame: {frame_idx}. Track {track_idx} killed.')

            for face_idx in faces_unassigned:
                new_track = Track(faces[face_idx])
                tracks_alive.append(new_track)
                logger.debug(f'Frame: {frame_idx}. New track {len(tracks_alive)} generated.')

            prev_frame_idx = frame_idx

        for track in tracks_alive:
            tracks_dead.append(track)

        return tracks_dead


def moving_crop(clip, fx, fy, size=(160, 160)):
    def scale(im, nR, nC):
        nR0 = len(im)  # source number of rows
        nC0 = len(im[0])  # source number of columns

        im = np.asarray([[im[int(nR0 * r / nR), int(nC0 * c / nC)]
                          for c in range(nC)] for r in range(nR)])

        return im

    def fl(gf, t):
        im = gf(t)

        x1, x2 = list(map(int, fx(int(t * 25))))
        y1, y2 = list(map(int, fy(int(t * 25))))

        im = im[y1:y2, x1:x2].copy()
        return scale(im, size[0], size[1])

    return clip.fl(fl)
