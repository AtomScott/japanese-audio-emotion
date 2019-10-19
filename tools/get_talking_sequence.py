from arguments import parse_args
from logger import create_logger
from tests import Tester

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

def download_images(query, n_images, out_dir_path, chromedriver_path="./chromedriver"):
    """Downloads face images for a query.
    
    Arguments:
        query {str} -- Query to download images for.
        n_images {int} -- Number of images to download.
        out_dir_path {str} -- Output path for images.
    
    Keyword Arguments:
        chromedriver_path {str} -- Path to chromedriver (default: {"./chromedriver"})
    
    Returns:
        out_image_paths {list} -- list of absolute paths to images
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
        out_image_paths = download_images(query, n_images, out_dir_path, chromedriver_path)

    return out_image_paths

# TODO: fix docstring
# TODO: return image paths as well if necessary 
def embed_faces(in_image_paths, save_embeddings=True, image_size=160, replace_images=False):
    """Crops face(faces) in image and return the cropped area(areas) along with an embedding(embeddings).

    Arguments:
        image_paths {list} -- list of paths to images

    Keyword Arguments:
        save {bool} -- If True, saves the cropped faces and embeddings to a file and returns the out_paths. (default: {False})

    Returns:
       embeddings, faces {[ndarray], [ndarray]} -- if save==False
       embedding_paths, face_paths {[str], [str]} -- if save==True
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


if __name__ =="__main__":
    args = parse_args()
    logger = create_logger()
    tester = Tester(args.debug)

    #############
    # * Set paths
    #############
    image_dir_path = os.path.join(args.out_dir_path, 'images', args.query)
    embedding_dir_path = os.path.join(args.out_dir_path, 'embeddings', args.query)
    if args.overwrite and os.path.isdir(args.out_dir_path):
        shutil.rmtree(args.out_dir_path)
    os.makedirs(image_dir_path, exist_ok=True)
    os.makedirs(embedding_dir_path, exist_ok=True)

    ##################
    # * Scrape images
    ##################
    scraped_image_paths = download_images(args.query, args.n_images, os.path.join(args.out_dir_path, 'images'), args.chromedriver_path)

    test = lambda x: 0 < len(x) <= args.n_images
    tester.test_list(scraped_image_paths, test)
    logger.info("Scraped {0} images of {1}".format(len(scraped_image_paths), args.query))

    ###############
    # * Embed faces
    ###############
    embeddings, faces, boxes = embed_faces(scraped_image_paths, save_embeddings=True, image_size=160, replace_images=True)

    test = lambda x: 0 < len(x[0]) == len(x[1]) and len(x[0]) == len(x[2])
    tester.test_list([embeddings, faces, boxes], test)
    logger.info('Generated {0} facial embeddings for {1}'.format(len(embeddings), args.query))

    #####################
    # * Outlier Detection
    #####################
    embeddings = detect_outliers(embeddings)

    ##################
    # * Scrape videos
    #################

