from tempfile import tempdirectory

from JAVER.arguments import parse_args
from JAVER.scrape_tools import scrape_videos, scrape_images

from JAVER.utils.outliers import remove_outlier_faces
from JAVER.utils.logging import create_logger
from JAVER.utils.arguments import parse_args

if __name__ =="__main__":

    args = parse_args()
    logger = create_logger()
    queries = args.queries

    for query in queries:
        with tempfile.TemporaryDirectory() as tmp_dir:

            #############
            # * Set paths
            #############
            image_dir_path = os.path.join(tmp_dir, 'images')
            video_dir_path = os.path.join(tmp_dir, 'videos')
            embedding_dir_path = os.path.join(tmp_dir, 'embeddings')

            os.makedirs(image_dir_path)
            os.makedirs(video_dir_path)
            os.makedirs(embedding_dir_path)

            ##################
            # * Scrape images
            ##################
            image_paths = scrape_images(
                query, 
                args.n_images,
                image_dir_path,
                args.chromedriver_path
                )
            
            #################
            # * Scrape videos
            #################
            video_paths = scrape_videos(
                args.query,
                video_dir_path,
                args.n_videos
                )

            logger.info("Scraped {0} images of {1}".format(
                len(image_paths), args.query))

            #####################
            # * Clean reference data
            #####################
            _ = remove_outlier_faces(image_paths)


            ######################
            # * Track targets face 
            ######################
            face_tracker = FaceTracker(
                image_size=image_size,
                ref_paths=ref_paths,
                batch_size=batch_size,
                step_large=step_large,
                step_small=step_small
            )

            for video_path in video_paths:
                face_tracker.track(video_path)
