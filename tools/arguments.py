import argparse

def parse_args():
    """Custom argument parser for command line usage.
    
    Returns:
        args {argparse.Namespace} -- dict of arguments
    """
    parser = argparse.ArgumentParser()

    # General
    parser.add_argument(
        '--debug',
        default=False,
        action='store_true',
        help='Outputs stuff for debugging'
    )

    #########################
    # * Args for Web Scraping
    #########################

    # General
    parser.add_argument(
        '--query', 
        type=str,
        default='Elon Musk',
        help='Search query')
    parser.add_argument(
        '--n_images',
        type=int,
        default=30,
        help='Number of images to download')
    parser.add_argument(
        '--n_videos',
        type=int,
        default=1,
        help='Number of images to download')
    parser.add_argument(
        '--out_dir_path',
        type=str,
        default='../datasets/XperFace/',
        help='path to image dir')
    parser.add_argument(
        '--chromedriver_path',
        type=str,
        default='./chromedriver',
        help='path to chrome_driver')
    parser.add_argument(
        '--overwrite',
        default=False,
        action='store_true',
        help='Overwrite output dir'
    )

    # Experimental


    #######################################
    # * Args for Facial Emotion Recognition
    #######################################

    # General

    # Experimental

    args = parser.parse_args()
    return args