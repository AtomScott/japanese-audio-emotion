import math
import os
import shutil
import pytest
import cv2
import numpy as np
from PIL import Image, ImageDraw


class TestFace:

    def test_is_valid(self, face):
        assert face.is_valid()

    def test_get_roi(self, face):
        dir = 'test_artifacts/TestFace'
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

        image = face.face_img
        image.save(os.path.join(dir, 'test_get_roi.jpeg'))

        assert image.width == 160
        assert image.height == 160

    def test_get_embedding(self):
        assert False


class TestTrajectory:
    def test_update_state(self):
        assert False

    def test_update_gallery(self):
        assert False


class TestFrameHandler:
    def test_next(self, log, frame_handler):
        dir = 'test_artifacts/TestFrameHandler'
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

        batch_size = frame_handler.batch_size
        step = frame_handler.step

        batch_count = 0
        while frame_handler.has_next():
            frames = frame_handler.next()
            for i, frame in enumerate(frames):
                im = Image.fromarray(frame)
                image_path = os.path.join(dir, f'{batch_size*batch_count*step + step*i}.png')
                im.save(image_path)
                log.debug(f'Image saved to {image_path}')

            batch_count += 1
            assert len(frames) == batch_size or len(frames) == len(frames) % batch_size

        # assert batch_count == math.ceil(len(frame_handler) / (step * batch_size))
        # assert frame_handler.idx == len(frame_handler) - len(frame_handler) % step

    def test_has_next(self):
        assert False


class TestFaceTracker:

    def test_track(self):
        assert False

    @pytest.mark.parametrize("video_path", [
        "data/Elon Musk/sample_short.mp4",
        # "data/Elon Musk/sample_mid.mp4",
    ])
    def test_detect(self, video_path, mtcnn, face_tracker, log):

        faces_dict = face_tracker.detect(video_path)

        n_faces_found = sum([len(faces) for _, faces in faces_dict.items()])
        assert n_faces_found >= 10  # There are def more than 10 faces in the sample movie

        log.info(faces_dict)
        log.info(n_faces_found)

        dir = os.path.join('test_artifacts/TestFaceTracker', os.path.basename(video_path))
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)

        for frame_index, faces in faces_dict.items():
            base_image = faces[0].img.copy()
            image_path = f'{frame_index}.jpeg'
            draw = ImageDraw.Draw(base_image)

            for i, face in enumerate(faces):
                draw.rectangle(face.bbox, outline=(255, 0, 255), width=2)
                # base_image = cv2.rectangle(base_image, face.bbox, (255, 0, 0), 2)

            base_image.save(os.path.join(dir, image_path))
            log.info(f'Saved to image {image_path}, n_faces: {len(faces)}')

    def test_detect_faces(self, face_tracker, multiple_images, log):
        images = multiple_images
        bboxes_found = face_tracker.detect_faces(images)

        assert len(bboxes_found) == len(images)
        assert all([bbox.size % 4 == 0 for bbox in bboxes_found])

        dir = [os.path.join('test_artifacts/TestFaceTracker/detect_faces', d) for d in ['original', 'faces']]
        for d in dir:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d)

        for i, image in enumerate(images):
            image.save(os.path.join(dir[0], f'{i}.png'))

        for i, bboxes in enumerate(bboxes_found):
            base_image = images[i].copy()
            draw = ImageDraw.Draw(base_image)
            for bbox in bboxes:
                draw.rectangle(list(map(int, bbox)), outline=(255, 0, 0), width=2)
            base_image.save(os.path.join(dir[1], f'{i}.png'))

        n_faces_found = sum([bbox.size / 4 for bbox in bboxes_found])
        if n_faces_found == 0:
            log.warning('No faces found!!!')
        else:
            log.info(f'Faces found == {n_faces_found}')

        log.info(f'bboxes_found.shape=={bboxes_found.shape}')
        log.info(f'bboxes_found.size=={bboxes_found.size}')




    def test_is_reference_face(self):
        assert False

    def test_associate(self):
        assert False
