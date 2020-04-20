import os
import shutil
from google_images_download import google_images_download

logger = create_logger()

def scrape_images(query, n_images, out_dir_path, chromedriver_path="./chromedriver"):
    """Downloads face images for a query.
    
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
