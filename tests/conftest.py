from pathlib import Path

import colorlog
import imageio
import numpy as np
import pytest
from PIL import Image


@pytest.fixture()
def log():
    handler = colorlog.StreamHandler()

    fmt_str = '%(log_color)s %(asctime)s%(levelno)s @%(funcName)s:%(lineno)d:\t%(message)s'

    colors = {
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }

    handler.setFormatter(
        colorlog.ColoredFormatter(fmt_str, '%H:%M:%S', log_colors=colors))

    logger = colorlog.getLogger('example')

    logger.addHandler(handler)
    logger.setLevel('DEBUG')

    return logger


@pytest.fixture()
def single_image() -> np.ndarray:
    img = Image.open("data/sample_image_1.jpeg")
    arr = np.asarray(img)
    return arr


@pytest.fixture()
def multiple_images():
    image_paths = Path(".").glob("data/Elon Musk/images/*")
    images = []
    for image_path in image_paths:
        try:
            image = Image.open(image_path)
            images.append(image.resize((500, 500)))
        except OSError as e:
            pass
    return images


@pytest.fixture()
def video_data():
    video_path = "data/Elon Musk/cropped_video.mp4"
    reader = imageio.get_reader(video_path, 'ffmpeg')
    return reader


@pytest.fixture()
def frame_handler():
    from JAVER.track_tools import FrameHandler
    step = 5
    batch_size = 25
    video_path = "data/Elon Musk/sample_short.mp4"
    return FrameHandler(step, batch_size, video_path)


@pytest.fixture()
def face_tracker():
    from JAVER.track_tools import FaceTracker

    p = Path("data/Elon Musk/")
    ref_paths = list(p.glob("inliers/*"))
    face_tracker = FaceTracker(image_size=160, ref_paths=ref_paths, batch_size=25, step_large=30, step_small=5)
    return face_tracker


@pytest.fixture()
def mtcnn():
    import torch
    from facenet_pytorch import MTCNN
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(image_size=160, keep_all=True, device=device)
    return mtcnn


@pytest.fixture()
def face():
    from JAVER.track_tools import Face
    image_path = "./data/cropped/sample_image_1_0.jpg"
    image = Image.open(image_path)
    face = Face(idx=0, bbox=(0, 0, image.width, image.height), img=image)
    return face
