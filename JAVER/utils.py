import os, shutil, csv
import random, string
from collections import defaultdict

from PIL import Image
import torch
import numpy as np


from facenet_pytorch import MTCNN, InceptionResnetV1
from facenet_pytorch.models import utils as facenet_utils
from facenet_pytorch.models.mtcnn import prewhiten

# import warnings
# with warnings.catch_warnings():
#     warnings.simplefilter("ignore")
from pyod.models.ocsvm import OCSVM

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

def crop_faces(in_image_paths=[], image_size=160, replace_images=False, threshold=0.98, out_paths = [], return_values=['bboxes', 'out_paths', 'rois'], images=None):    
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
    logger.info('using: {0}'.format('cuda:0' if torch.cuda.is_available() else 'cpu'))
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
            logger.debug('Skipping bad image {0}. Error: {1}'.format(image_path, e))
            continue

        if boxes is None: 
            continue

        for i, box in enumerate(boxes):
            if probs[i]>=threshold:
                # ? Not sure why this is necessary..
                # ? but sometimes breaks without
                x1,y1,x2,y2 = box
                box = (
                    min(x1, x2),# left
                    min(y1, y2),# lower
                    max(x1, x2),# right
                    max(y1, y2) # upper
                    )

                roi = image.crop(box).resize((image_size,image_size))
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
    logger.info('using: {0}'.format('cuda:0' if torch.cuda.is_available() else 'cpu'))
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    ret_dct = defaultdict(lambda : [])
    if not in_image_paths:
        in_image_paths = ['' for _ in images]
    if not out_paths:
        out_paths = ['' for _ in in_image_paths]
    assert len(in_image_paths) == len(out_paths)

    for i, image_path in enumerate(in_image_paths):
        out_path = out_paths[i]
        # try :
        image = Image.open(image_path) if image_path!='' else images[i]

        face = np.array(image, dtype=np.float32)
        face = face.transpose((2,0,1))
        face = prewhiten(torch.tensor(face))
        aligned = torch.stack([face]).to(device) # ? 
        embedding = resnet(aligned).detach().cpu()

        if out_path:
            try:
                np.save(out_path, embedding)
            except OSError as e:
                basename = os.path.basename(out_path)
                fname, ext = os.path.splitext(basename)

                letters = string.ascii_lowercase
                random_string = ''.join(random.choice(letters) for i in range(16))
                shortened = out_path.replace(fname, random_string)
                np.save(shortened, embedding)

                logger.warning('{0}\nGenerated new path {1}'.format(e, shortened))

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
        y = clf.predict(data.reshape(1,-1))
        if y: # y==1 for outliers
            logger.debug('Found outlier: {0}'.format(index))
            outlier_idx.append(index)
        else:
            inlier_idx.append(index)

    logger.info('{:.0%} are outliers'.format(len(outlier_idx) / len(lst)))
    return inlier_idx, outlier_idx

