from code.arguments import parse_args
from code.logger import create_logger
from code.tests import Tester

from google_images_download import google_images_download
import os
import shutil

from facenet_pytorch import MTCNN, InceptionResnetV1
from facenet_pytorch.models import utils as facenet_utils
from facenet_pytorch.models.mtcnn import prewhiten
from PIL import Image
import torch
import numpy as np

from pyod.models.auto_encoder import AutoEncoder

from pytube import YouTube
from bs4 import BeautifulSoup as bs
import requests, re

import imageio  
from tqdm import tqdm
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


def download_images(query, n_images, out_dir_path, chromedriver_path="./chromedriver"):
    """Downloads images of a given query from google image search.
    
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
    """
    assert os.path.isfile(chromedriver_path), logger.critical("chromedriver not found")
    assert os.access(chromedriver_path, os.X_OK), logger.critical("chromedriver has incorrect permissions")

    response = google_images_download.googleimagesdownload()

    args = {
        "keywords": query,
        "limit": n_images,
        "format": "jpg",
        "type": "face",
        "output_directory": out_dir_path,
        "chromedriver": "./chromedriver",
        "silent_mode": True
    }
    out_image_paths = response.download(args)[0][query]
    if len(out_image_paths) == 0:
        logger.warning("Error. Couldn't get images, trying again.")
        out_image_paths = download_images(query, n_images, out_dir_path, chromedriver_path)

    return out_image_paths

# TODO: fix docstring
# TODO: return image paths as well if necessary 
def embed_faces(in_image_paths, save_embeddings=True, image_size=160, replace_images=False):
    """Crops face(faces) in image and return the cropped area(areas) along with an embedding(embeddings).

    
    
    Parameters
    ----------
    in_image_paths : list
        Path to images to crop
    save_embeddings : bool, optional
        Save the embeddings, by default True
    image_size : int, optional
        [description], by default 160
    replace_images : bool, optional
        [description], by default False
    
    Returns
    -------
    [type]
        [description]
    """    

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(image_size=image_size, keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    
    all_embeddings = []
    all_faces = []
    all_boxes = []
    for image_path in in_image_paths:
        try :
            image = Image.open(image_path)
            boxes, _ = mtcnn.detect(image)

            for index, box in enumerate(boxes):
                if replace_images:
                    os.remove(image_path)
                    face = facenet_utils.detect_face.extract_face(image, box=box, save_path=image_path, image_size=image_size)
                else:
                    face = facenet_utils.detect_face.extract_face(image, box=box, image_size=image_size)
                
                face = prewhiten(face)
                aligned = torch.stack([face]).to(device)
                embedding = resnet(aligned).detach().cpu()


                if save_embeddings is not None:
                    dir_path, file_name = os.path.split(image_path)
                    fname, _ = os.path.splitext(file_name)
                    out_embedding_path = os.path.join(dir_path.replace('images', 'embeddings'), fname+str(index)+'.npy')
                    np.save(out_embedding_path, embedding)  
            
                all_embeddings.append(embedding.cpu().detach().numpy()[0])
                all_faces.append(face)
                all_boxes.append(box)

        except Exception as e:
            logger.warning('Bad Image: {0}. Skipping..'.format(e))
    
    return all_embeddings, all_faces, all_boxes


# TODO: add docstring
def detect_outliers(lst):
    clf = AutoEncoder(verbose=1)
    clf.fit(lst)
    
    inliers = []
    for index, data in enumerate(lst):
        y = clf.predict(data.reshape(1,-1))
        if y: # y==1 for outliers
            logger.warning('Found outlier: {0}'.format(index))
        else:
            inliers.append(data)

    logger.info('{:.0%} are outliers'.format(1 - len(inliers) / len(lst)))
    return inliers

# add docstring
def scrape_videos(query, out_path, n):
    url ='https://www.youtube.com/results?search_query='+query
    r = requests.get(url)
    page = r.text
    soup=bs(page,'html.parser')
    res=soup.find_all('a',{'href': re.compile(r'watch')})
    i = 0
    out_file_paths = []
    links = []
    for l in res:
        fname = '{0}.mp4'.format(i)
        link = "https://www.youtube.com"+l.get("href")
        if link not in links:
            links.append(link)
            myVideo = YouTube(link)
            print(out_path, str(i))
            myVideo.streams.first().download(output_path=out_path, filename=str(i))
            i += 1 
            out_file_paths.append(os.path.join(out_path, fname))
            logger.info("Downloaded video: {0} @ {1}".format(link, os.path.join(out_path, fname)))
        else:
            pass
        if i >= n:
            break
    return out_file_paths

    


        

        

    
    
