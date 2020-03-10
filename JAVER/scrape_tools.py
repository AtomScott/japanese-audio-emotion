"""Tools to scrape the internet for data
"""

import os, sys, shutil, requests
import re

import atomity

from google_images_download import google_images_download
from .logger import create_logger

from pytube import YouTube
from bs4 import BeautifulSoup as bs

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
    assert os.path.isfile(chromedriver_path), logger.critical("chromedriver not found")
    assert os.access(chromedriver_path, os.X_OK), logger.critical("chromedriver has incorrect permissions")

    response = google_images_download.googleimagesdownload()

    args = {
        "keywords": query,
        "limit": n_images,
        "format": "jpg",
        "type": "face",
        "output_directory": out_dir_path,
        "chromedriver": chromedriver_path,
        "silent_mode": True,
        "verbose":True
    }
    with atomity.suppress_stdout():
        out_image_paths = response.download(args)[0][query]
    if len(out_image_paths) == 0:
        logger.debug("Error. Couldn't get images, trying again.")
        out_image_paths = get_face_images(query, n_images, out_dir_path, chromedriver_path)
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
        logger.debug("Trying download for {0}".format(link))
        if link not in links:
            links.append(link)
            try:
                myVideo = YouTube(link)
                myVideo.streams.first().download(output_path=os.path.join(out_dir_path, query), filename=str(i))
                out_file_paths.append(os.path.join(out_dir_path, fname))
                logger.debug("Downloaded video: {0} @ {1}".format(link, os.path.join(out_dir_path,query, fname)))
                i += 1 
            except Exception as e:
                logger.warning(e)
        else:
            pass
        if i >= n:
            break
    logger.info("Successfully downloaded {0} videos.".format(i))
    return out_file_paths
