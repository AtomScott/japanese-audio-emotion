import argparse
import sys

import cv2

from create_image_dataset import *
from outlier_detection import *

if __name__ == "__main__":

    # Read names in namelist.txt as well as removing \n
    name_list = open("name_list.txt", "r").read().splitlines()

    """CREATE IMAGE DATASET"""
    # Scrape images from google
    dir = "../data/images/"
    for line in name_list:
        query, yt_url = line.split(" ")
        path = dir + query + "/scrapped"
        scrape_images(query, path, 128)
    print("[COMPLETED] Finished Scraping Images")

    image_paths = list(paths.list_images(dir))  # List all images within dir

    n = 0  # Used to calculate average_landmarks online, needed to warp align faces
    average_landmarks = [[0, 0]] * 72  # Init average_landmarks to a 2d zero array

    path_dict = {}  # helpful to have a dict to remember path/roi/landmarks
    for (i, image_path) in enumerate(image_paths):
        image = cv2.imread(image_path)
        name = image_path.split("/")[-3]

        # Apply deep learning-based face detector to localize faces in the input image
        try:
            face_landmarks = face_recognition.face_landmarks(image, model="large")
        except:
            os.remove(image_path)

        if len(face_landmarks) > 0:  # ensure at least one face was found
            tri_list, pt_list, delaunay_face = get_delaunay(face_landmarks, image)
            ROIs, pt_list2 = tilt_crop_scale(pt_list, image)

            # update average landmarks and image_dict
            for points, roi in zip(pt_list2, ROIs):
                average_landmarks += points

                # Save ROI, landmarks
                roi_path = dir + name + "/roi/"
                landmarks_path = dir + name + "/landmarks/"
                embedding_path = dir + name + "/embedding/"

                for path in [roi_path, landmarks_path, embedding_path]:
                    if not os.path.exists(path):
                        os.makedirs(path)

                # Add path to file to image_dict
                path_dict[name + "_" + str(n)] = {
                    "ROI": roi_path + str(n) + ".png",
                    "LANDMARKS": landmarks_path + str(n) + ".npy",
                    "EMBEDDING": embedding_path + str(n) + ".npy",
                }
                cv2.imwrite(path_dict[name + "_" + str(n)]["ROI"], roi)
                np.save(path_dict[name + "_" + str(n)]["LANDMARKS"], points)

                print("[IN PROCESS] Added {0} to image dict".format(name + str(n)))
                n += 1

    print("[COMPLETED] {0} Faces Were Recognized".format(n))

    w = 224
    h = 224
    peripheral_points = [
        [0, 0],
        [0, w / 2],
        [0, w - 1],
        [h / 2, 0],
        [h / 2, w - 1],
        [h - 1, 0],
        [h - 1, w / 2],
        [h - 1, w - 1],
    ]
    # Calculate average landmarks and add peripheral points.
    average_landmarks /= n
    average_landmarks = np.append(average_landmarks, peripheral_points, axis=0)
    np.save("../data/average_landmarks.npy", average_landmarks)
    print("[COMPLETED] Finished Average Landmark Calculation")

    # Face Warp
    # face_dict contains ROI, EMBEDDING and LANDMARKS
    del_list = []
    for key, face_dict in path_dict.items():
        fname = "{0}.jpg".format(key.split("_")[1])
        np.save(
            face_dict["LANDMARKS"],
            np.append(np.load(face_dict["LANDMARKS"]), peripheral_points, axis=0),
        )
        warp_face(face_dict["ROI"], face_dict, average_landmarks)
        print("[IN PROCESS] Saved {0}".format(key))

        # create facial embeddings for each ROI
        ROI = cv2.imread(face_dict["ROI"])
        ROI = cv2.cvtColor(ROI, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(ROI)
        np.save(path_dict[key]["EMBEDDING"], enc)
        if not enc:
            os.remove(face_dict["ROI"])
            os.remove(face_dict["LANDMARKS"])
            os.remove(face_dict["EMBEDDING"])
            del_list.append(key)

    for key in del_list:
        del path_dict[key]
    np.save("../data/path_dict.npy", path_dict)

    # Outlier Detection
    path_dict = purify(path_dict, name_list)
    np.save("../data/path_dict.npy", path_dict)

    print("[Completed] Created Image Dataset")
