import argparse
import csv
import glob
import os
import random
import re
import shutil
import string
import sys
from collections import defaultdict

import atomity
import chainer
import chainer.functions as F
import chainer.links as L
import imageio
import matplotlib.pyplot as plt
import more_itertools as mit
import numpy as np
import requests
import torch
from bs4 import BeautifulSoup as bs
from chainer.training import extensions
from facenet_pytorch import MTCNN, InceptionResnetV1
from facenet_pytorch.models import utils as facenet_utils
from facenet_pytorch.models.mtcnn import prewhiten
from google_images_download import google_images_download
from PIL import Image
# import warnings
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
from pyod.models.ocsvm import OCSVM
from pytube import YouTube
from sklearn import datasets, svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.multiclass import unique_labels
from tqdm import tqdm

import japanize_matplotlib
from arguments import parse_args
from datasets.dataset import ESDataset
from models.cnn import CNN
from utils import create_result_dir

from .logger import create_logger

logger = create_logger(name=__name__)


def make_save_dirs(out_dir_path, dirs1, dirs2, overwrite=False):
    """Make directories to save images, videos and embeddings for a query

    The directories saved will be structured s.t.
    .
    +-- out_dir_path
    |   +-- images
    |       +-- query
    |   +-- videosunction to rea
    |       +-- query
    |   +-- embeddings
    |       +-- query
    |   +-- cropped_images
    |       +-- query
    |   +-- inliers
    |       +-- query
    |   +-- outliers
    |       +-- query


    Parameters
    ----------
    out_dir_path : str
        path to make new directories
    query : str
        [description]
    overwrite : bool, optional
        [description], by default False

    Warnings
    --------
    The overwrite parameter is destructive and will overwrite out_dir_path.

    TODO
    ----
    Consider the necessity of a utility function that creates a dir tree from a given dictionary or list of list? maybe it exists already.
    """
    if overwrite and os.path.isdir(out_dir_path):
        shutil.rmtree(out_dir_path)

    for dir2 in dirs2:
        for dir1 in dirs1:
            path = os.path.join(out_dir_path, dir1, dir2)
            os.makedirs(path, exist_ok=True)


def read_querylist(path):
    """Utility function to read first column of csv.

    Excludes the first row because it is usually a header.

    Parameters
    ----------
    path : str
        path to csv

    """
    queries = []
    with open(path, 'r') as csvfile:
        # Creating a csv reader object
        reader = csv.reader(csvfile)

        # Extracting field names in the first row
        fields = next(reader)

        # Extracting each data row one by one
        for row in reader:
            queries.append(row[0])

        return queries


def crop_faces(in_image_paths=[], image_size=160, replace_images=False, threshold=0.98, out_paths=[], return_values=['bboxes', 'out_paths', 'rois'], images=None):
    """Crops faces in image to a given size. 

    Parameters
    ----------
    in_image_paths : list
        Path to images to crop.
    image_size : int, optional
        [description], by default 160
    replace_value : bool, optional
        Replace the input images by the cropped images, by default False
    return_values : list, optional
        [description], by default ['bboxes', 'out_paths', 'rois', 'landmarks']

    Returns
    -------
     : dict
        A dictionary containing values included in `return_value`. 

    TODO
    ----
    Alignment functionality
        It should be easy to align the faces using the detected facial landmarks. Might be a good preprocessing step.
    """

    ret_dct = {key: [] for key in return_values}
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info('using: {0}'.format(
        'cuda:0' if torch.cuda.is_available() else 'cpu'))
    mtcnn = MTCNN(image_size=image_size, keep_all=True, device=device)

    if not in_image_paths:
        in_image_paths = ['' for _ in images]
    if not out_paths:
        out_paths = ['' for _ in in_image_paths]
    assert len(in_image_paths) == len(out_paths)

    for i, image_path in enumerate(in_image_paths):
        out_path = out_paths[i]
        logger.debug('Detecting faces in {0}'.format(image_path))
        try:
            image = Image.open(image_path) if image_path != '' else images[i]
            logger.debug('image shape = {0}'.format(np.array(image).shape))
            boxes, probs = mtcnn.detect(image)
        except Exception as e:
            logger.debug(
                'Skipping bad image {0}. Error: {1}'.format(image_path, e))
            continue

        if boxes is None:
            continue

        for i, box in enumerate(boxes):
            if probs[i] >= threshold:
                # ? Not sure why this is necessary..
                # ? but sometimes breaks without
                x1, y1, x2, y2 = box
                box = (
                    min(x1, x2),  # left
                    min(y1, y2),  # lower
                    max(x1, x2),  # right
                    max(y1, y2)  # upper
                )

                roi = image.crop(box).resize((image_size, image_size))
                if 'bboxes' in return_values:
                    ret_dct['bboxes'].append(box)

                if 'out_paths' in return_values:
                    base, name = os.path.split(out_path)
                    name = '{0}_{1}.jpg'.format(os.path.splitext(name)[0], i)
                    save_path = os.path.join(base, name)
                    roi.save(save_path)
                    ret_dct['out_paths'].append(save_path)

                if 'rois' in return_values:
                    ret_dct['rois'].append(roi)

                # if 'landmarks' in return_values:
                #     ret_dct['landmarks'].append(landmarks[i])

    return ret_dct


def embed_faces(in_image_paths=[], out_paths=[], return_values=['embeddings', 'out_paths'], images=None):
    """Crops faces) in image and return the cropped area(areas) along with an embedding(embeddings).

    Parameters
    ----------
    in_image_paths : list
        Path to images to crop
    save_embeddings : bool, optional
        Save the embeddings, by default True
    image_size : int, optional
        [description] , by default 160
    replace_images : bool, optional
        [description], by default False

    Returns
    -------
    all_embeddings : list
        A list of embeddings for each face. Each embeddings is of size (512). We use IneptionResnetV1 pretrained on vggface2.
    all_faces : list
        A list of ROIs for each face. 
    all_boxes : list
        A list of bboxes for each face

    Warning
    -------
    Make sure that the input image only contains one face!
    """

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logger.info('using: {0}'.format(
        'cuda:0' if torch.cuda.is_available() else 'cpu'))
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    ret_dct = defaultdict(lambda: [])
    if not in_image_paths:
        in_image_paths = ['' for _ in images]
    if not out_paths:
        out_paths = ['' for _ in in_image_paths]
    assert len(in_image_paths) == len(out_paths)

    for i, image_path in enumerate(in_image_paths):
        out_path = out_paths[i]
        # try :
        image = Image.open(image_path) if image_path != '' else images[i]

        face = np.array(image, dtype=np.float32)
        face = face.transpose((2, 0, 1))
        face = prewhiten(torch.tensor(face))
        aligned = torch.stack([face]).to(device)  # ?
        embedding = resnet(aligned).detach().cpu()

        if out_path:
            try:
                np.save(out_path, embedding)
            except OSError as e:
                basename = os.path.basename(out_path)
                fname, ext = os.path.splitext(basename)

                letters = string.ascii_lowercase
                random_string = ''.join(random.choice(letters)
                                        for i in range(16))
                shortened = out_path.replace(fname, random_string)
                np.save(shortened, embedding)

                logger.warning(
                    '{0}\nGenerated new path {1}'.format(e, shortened))

        if 'out_paths' in return_values:
            ret_dct['out_paths'].append(out_path)

        if 'embeddings' in return_values:
            ret_dct['embeddings'].append(embedding)

        # except Exception as e:
        #     logger.warning('Bad Image: {0}. Skipping..'.format(e))

    return ret_dct


def detect_outliers(lst):
    """detect outliers in a list of numpy arrays

    Parameters
    ----------
    lst : List
        [description]

    Returns
    -------
    inliers : List
        A list of the inliers
    """
    clf = OCSVM(verbose=0)
    clf.fit(lst)

    inlier_idx = []
    outlier_idx = []
    for index, data in enumerate(lst):
        y = clf.predict(data.reshape(1, -1))
        if y:  # y==1 for outliers
            logger.debug('Found outlier: {0}'.format(index))
            outlier_idx.append(index)
        else:
            inlier_idx.append(index)

    logger.info('{:.0%} are outliers'.format(len(outlier_idx) / len(lst)))
    return inlier_idx, outlier_idx


"""Tools to scrape the internet for data
"""


logger = create_logger(name=__name__)


def get_face_images(query, n_images, out_dir_path, chromedriver_path="./chromedriver"):
    """Downloads face images for a query. 

    This wraps [google_images_download](https://google-images-download.readthedocs.io/en/latest/arguments.html), but constrained to type=faces so that only images of faces are scraped. We rely on [type setting](https://www.google.com/advanced_image_search) in Google Image Search to find face images.

    Parameters
    ----------
    query : str
        Query to download images for.
    n_images : int
        Number of images to download.
    out_dir_path : str
        Output path for images.
    chromedriver_path : str
        Path to chromedriver
    chromedriver_path : str
        Path to chromedriver (default: {"./chromedriver"})

    Returns
    -------
    out_image_paths : list
        list of absolute paths to images

    Examples
    --------

    """
    assert os.path.isfile(chromedriver_path), logger.critical(
        "chromedriver not found")
    assert os.access(chromedriver_path, os.X_OK), logger.critical(
        "chromedriver has incorrect permissions")

    response = google_images_download.googleimagesdownload()

    args = {
        "keywords": query,
        "limit": n_images,
        "format": "jpg",
        "type": "face",
        "output_directory": out_dir_path,
        "chromedriver": chromedriver_path,
        "silent_mode": True,
        "verbose": True
    }
    with atomity.suppress_stdout():
        out_image_paths = response.download(args)[0][query]
    if len(out_image_paths) == 0:
        logger.debug("Error. Couldn't get images, trying again.")
        out_image_paths = get_face_images(
            query, n_images, out_dir_path, chromedriver_path)
    else:
        logger.info(f'Success: Loaded {len(out_image_paths)} images.')
    return out_image_paths

# TODO: Make sure that videos are downloaded!!


def get_yt_videos(query, out_dir_path, n):
    """Downloads youtube videos from a given query.

    Parameters
    ----------
    query : str
        The query to search for
    out_dir_path : str
        path to save the videos
    n : int
        Number of videos to save

    Returns
    -------
    out_paths : list
        list of absolute paths to the videos
    """

    url = 'https://www.youtube.com/results?search_query='+query
    r = requests.get(url)
    page = r.text
    soup = bs(page, 'html.parser')
    res = soup.find_all('a', {'href': re.compile(r'watch')})
    i = 0
    out_file_paths = []
    links = []

    for l in res:
        fname = '{0}.mp4'.format(i)
        link = "https://www.youtube.com"+l.get("href")
        logger.debug("Trying download for {0}".format(link))
        if link not in links:
            links.append(link)
            try:
                myVideo = YouTube(link)
                myVideo.streams.first().download(
                    output_path=os.path.join(out_dir_path, query), filename=str(i))
                out_file_paths.append(os.path.join(out_dir_path, fname))
                logger.debug("Downloaded video: {0} @ {1}".format(
                    link, os.path.join(out_dir_path, query, fname)))
                i += 1
            except Exception as e:
                logger.warning(e)
        else:
            pass
        if i >= n:
            break
    logger.info("Successfully downloaded {0} videos.".format(i))
    return out_file_paths


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
            first_true = mit.first_true(
                probs, default=-1, pred=lambda x: x is not None)
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


class CNN(chainer.Chain):
    def __init__(self):
        # クラスの初期化
        super(CNN, self).__init__(
            conv1=L.Convolution2D(1, 20, 4),  # フィルター3　ボケを作ってる、入力、出力
            conv2=L.Convolution2D(20, 30, 3),  # フィルター4
            conv3=L.Convolution2D(30, 40, 3),
            conv4=L.Convolution2D(40, 50, 3),

            l1=L.Linear(800, 500),
            l2=L.Linear(500, 500),
            l3=L.Linear(500, 10, initialW=np.zeros(
                (10, 500), dtype=np.float32))
        )

    def __call__(self, x):
        # 順伝播の計算を行う関数
        # :param x: 入力値
        h = F.max_pooling_2d(F.dropout(F.relu(self.conv1(x))), 2)
        h = F.max_pooling_2d(F.dropout(F.relu(self.conv2(h))), 2)
        h = F.max_pooling_2d(F.dropout(F.relu(self.conv3(h))), 2)
        h = F.dropout(F.relu(self.conv4(h)))

        h = F.relu(self.l1(h))
        h = F.relu(self.l2(h))
        y = self.l3(h)
        return y


sys.path.append('..')


def main(args):

    print(args.dataset_dir, os.path.isfile(args.dataset_dir))
    if os.path.isfile(args.dataset_dir):
        paths = [args.dataset_dir]
    else:
        paths = glob.glob(os.path.join(args.dataset_dir, '**/*.wav'))

    print(len(paths))
    train_paths, test_path = train_test_split(
        paths, train_size=0.9, test_size=0.1)
    train = ESDataset(train_paths, label_index=args.label_index)
    test = ESDataset(test_path, label_index=args.label_index)

    batchsize = args.batch_size
    train_iter = chainer.iterators.SerialIterator(train, batchsize)
    test_iter = chainer.iterators.SerialIterator(test, batchsize, False, False)

    gpu_id = 0  # Set to -1 if you use CPU

    model = CNN()
    if args.device >= 0:
        model.to_gpu(args.device)
    if args.init_weights != '':
        chainer.serializers.load_npz(args.init_weights, model)

    # Since we do not specify a loss function here, the default 'softmax_cross_entropy' is used.
    model = chainer.links.Classifier(model)

    # selection of your optimizing method
    optimizer = chainer.optimizers.Adam()

    # Give the optimizer a reference to the model
    optimizer.setup(model)

    # Get an updater that uses the Iterator and Optimizer
    updater = chainer.training.updaters.StandardUpdater(
        train_iter, optimizer, device=gpu_id)

    # Setup a Trainer
    trainer = chainer.training.Trainer(
        updater, (args.epoch, 'epoch'), out=args.out_dir)
    trainer.extend(extensions.Evaluator(test_iter, model, device=args.device))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ["epoch", "main/loss", "validation/main/loss", "main/accuracy", "validation/main/accuracy", "elapsed_time"]))
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'validation/main/loss'], x_key='epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(
        ['main/accuracy', 'validation/main/accuracy'], x_key='epoch', file_name='accuracy.png'))
    trainer.extend(extensions.snapshot_object(
        model.predictor, filename='model_epoch-{.updater.epoch}'), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.ProgressBar())

    trainer.run()

    chainer.serializers.save_npz(os.path.join(
        args.out_dir, 'weights.npz'), model)


def parse_args():
    parser = argparse.ArgumentParser(description='Training script')

    parser.add_argument(
        '--dataset_dir', default='./datasets/RAVDESS', type=str, help='path to dataset dir')
    parser.add_argument(
        '--out_dir', default='./results/', type=str, help='path to dataset dir')
    parser.add_argument(
        '--batch_size', default=128, type=int,
        help='batch size')
    parser.add_argument(
        '--device', default=-1, type=int,
        help='device to use cpu is -1')
    parser.add_argument(
        '--epoch', default=50, type=int,
        help='number of epochs to train')
    parser.add_argument(
        '--label_index', default=0, type=int,
        help='index of filename to use as a label')
    parser.add_argument(
        '--init_weights', default='', type=str,
        help='Add default weights to init the model.'
    )
    parser.add_argument(
        '--overwrite', default=False, action='store_true',
        help='overwrite out dir.'
    )
    parser.add_argument(
        '--title', default='Confusin Matrix', type=str,
        help='title for confusion matrix.'
    )

    args = parser.parse_args()
    return args


sys.path.append('..')


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    print(unique_labels(y_true, y_pred))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    im.set_clim(0, 1)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


def predict(model, x):
    y = np.argmax(model.predictor(x=np.array([x], dtype="float32")).data)
    return int(y)


def main(args):
    model = CNN()
    chainer.serializers.load_npz(args.init_weights, model)
    model = chainer.links.Classifier(model)

    paths = glob.glob(os.path.join(args.dataset_dir, '**/*.wav'))
    testset = ESDataset(paths, label_index=args.label_index)

    y_targs = []
    y_preds = []
    for data in testset:
        x, y = data

        y_targs.append(int(y))
        y_preds.append(int(predict(model, x)))

    class_names = ['中立', '穏やか', '幸せ', '悲しみ', '怒り', '恐怖', '嫌悪', '驚き']

    accuracy = accuracy_score(y_targs, y_preds)
    plot_confusion_matrix(y_targs, y_preds, classes=class_names, normalize=True,
                          title='{0} 精度:{1:.1%}'.format(args.title, accuracy))

    plt.savefig(os.path.join(args.out_dir, 'confusion_matrix.png'))


if __name__ == '__main__':
    args = parse_args()

    main(args)
