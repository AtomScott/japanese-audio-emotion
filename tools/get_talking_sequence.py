from arguments import parse_args
from logger import create_logger
from tests import Tester

from google_images_download import google_images_download
import os
import shutil



def download_images(query, n_images, out_dir_path, chromedriver_path="./chromedriver"):
    """Downloads face images for a query.
    
    Arguments:
        query {str} -- Query to download images for.
        n_images {int} -- Number of images to download.
        out_dir_path {str} -- Output path for images.
    
    Keyword Arguments:
        chromedriver_path {str} -- Path to chromedriver (default: {"./chromedriver"})
    
    Returns:
        [list] -- list of absolute paths to images
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


if __name__ =="__main__":
    args = parse_args()
    logger = create_logger()
    tester = Tester(args.debug)

    if args.overwrite and os.path.isdir(args.out_dir_path):
        shutil.rmtree(args.out_dir_path)
    os.makedirs(args.out_dir_path, exist_ok=True)

    scraped_image_paths = download_images(args.query, args.n_images, os.path.join(args.out_dir_path, 'images'), args.chromedriver_path)

    test = lambda x: 0 < len(x) <= args.n_images
    tester.test_list(scraped_image_paths, test)

