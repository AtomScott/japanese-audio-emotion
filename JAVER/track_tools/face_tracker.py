"""

"""

import numpy as np

from PIL import Image

from cvlab_toolbox.models import SubspaceMethod


class FaceTracker:
    """
    Track faces of a single person for a given video.

        Example
       --------
        >>> face_tracker = FaceTracker(160, "Elon_img.png", 1, 3, 1)
        >>> face_tracker.track("Elon_vid.mp4")

    """

    def __init__(self, image_size: int, ref_paths:list, batch_size: int, step_large: int, step_small:int):
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

            ###############
            # * Save tracks
            ###############

            clip = VideoFileClip(video_path)
            for track in tracks:
                x1s, y1s, x2s, y2s = track.get_bboxes()
                def fx(t): return (x1s[t], x2s[t])
                def fy(t): return (y1s[t], y2s[t])
                clip = clip.fx(moving_crop, fx=fx, fy=fy)
        return None

    def detect(self, video_path:str)->dict:
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

    def detect_faces(self, images:PIL.Image, threshold:float=0.5)->numpy.ndarray:
        """Detect faces using an MTCNN detector
        
        Parameters
        ----------
        images : PIL.Image
            [description]
        threshold : float, optional
            [description], by default 0.5
        
        Returns
        -------
        numpy.ndarray
            A 2d list containing bounding boxes for each frame.
        """
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

    def is_reference_face(self, face):
        if face.is_valid():
            probs = self.clf.predict_proba(face.embedding).squeeze()
            return probs > 0.5
        else:
            return False

    def associate(self, faces_dict):

        salt = 1 / 10 ** 9
        pepper = 10 ** 6

        tracks_alive = []
        tracks_dead = []

        lam = 0.1
        thresh_1, thresh_2 = 0.6, 0.3
        kill_thresh = 2

        for _, (frame, faces) in enumerate(faces_dict.items()):
            if _:  # except first case

                C = association_cost_matrix(faces, tracks_alive, lam)
                G = gate_matrix(faces, tracks_alive, thresh_1, thresh_2) + salt
                gated_cost_matrix = C / G
                assert C.shape == G.shape, f'{C.shape}, {G.shape}'
                row_idxs, col_idxs = linear_sum_assignment(gated_cost_matrix)

                # row_idxs[i] is assigned to col_idxs[j]
                # (faces[ri] is assigned to tracks_alive[ci])
                for ri, ci in zip(row_idxs, col_idxs):
                    if gated_cost_matrix[ri, ci] < pepper:
                        logger.debug(
                            f'Frame: {frame}. Add face {ri} to track {ci}')
                        tracks_alive[ci].update(faces[ri])
                        del faces[ri]
                    else:
                        tracks_alive[ci].update(None)
                        logger.debug(
                            f'Frame: {frame}. Track {ci} was not associated to any face. none_count={tracks_alive[ci].none_count}')

                for i, track in enumerate(tracks_alive):
                    if i not in col_idxs:
                        tracks_alive[i].update(None)
                        logger.debug(
                            f'Frame: {frame}. Track {ci} was not associated to any face. none_count={tracks_alive[ci].none_count}')

                    # Terminate track with continous null updates
                    if track.none_count >= kill_thresh:
                        tracks_dead.append(track)
                        del tracks_alive[i]
                        logger.debug(f'Frame: {frame}. Track {i} killed.')

            # Generate new tracks if required
            for face in faces:
                new_track = Track(face)
                tracks_alive.append(new_track)

        for track in tracks_alive:
            tracks_dead.append(track)

        return tracks_dead


def moving_crop(clip, fx, fy, size=(160, 160)):
    def scale(im, nR, nC):
        nR0 = len(im)  # source number of rows
        nC0 = len(im[0])  # source number of columns
        return np.asarray([[im[int(nR0 * r / nR)][int(nC0 * c / nC)]
                            for c in range(nC)] for r in range(nR)])

    def fl(gf, t):
        im = gf(t)

        x1, x2 = list(map(int, fx(int(t * 25))))
        y1, y2 = list(map(int, fy(int(t * 25))))

        return scale(im[y1:y2, x1:x2], 160, 160)

    return clip.fl(fl)
