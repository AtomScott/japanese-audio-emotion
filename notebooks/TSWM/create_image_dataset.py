import argparse
import math
import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import requests
from bs4 import BeautifulSoup as bs
from PIL import Image
from scipy.spatial import Delaunay

import cv2
import face_recognition
import imutils
from google_images_download import google_images_download
from imutils import paths


def scrape_images(query, path, n):
    """
    desc:
        scrape all images for a query retriveable from a google image search.
    args:
        1st -> google image query
        2nd -> folder to save images
        3rd -> number of images to save
    """
    response = google_images_download.googleimagesdownload()  # class instantiation
    if not os.path.exists(path):
        os.makedirs(path)
    arguments = {
        "keywords": query + " youtuber",
        "limit": n,
        "print_urls": False,
        "output_directory": "./",
        "image_directory": path,
        "chromedriver": "../chromedriver",
    }  # creating list of arguments
    paths = response.download(arguments)  # passing the arguments to the function


def unit_vector(vector):
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def show_img(img):
    plt.axis("off")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()


def get_landmarks(faces, image):
    dst = image.copy()
    for landmarks in faces:
        for facial_area in landmarks:
            for (x, y) in landmarks[facial_area]:
                dst = cv2.circle(dst, (x, y), 2, (0, 0, 255), 2)
    return dst


def get_delaunay(faces, image):
    dst = image.copy()
    tri_list = []
    pt_list = []
    for landmarks in faces:
        points = []
        for facial_area in landmarks:
            for (x, y) in landmarks[facial_area]:
                points.append([x, y])
        points = np.asarray(points)
        tri = Delaunay(points)
        for indice in tri.simplices:
            x1, x2, x3 = points[indice].T[0]
            y1, y2, y3 = points[indice].T[1]
            cv2.line(dst, (x1, y1), (x2, y2), (255, 0, 0))
            cv2.line(dst, (x2, y2), (x3, y3), (255, 0, 0))
            cv2.line(dst, (x3, y3), (x1, y1), (255, 0, 0))
        tri_list.append(tri)
        pt_list.append(points)
    return tri_list, pt_list, dst


def align_face(face_landmark, image, tri):
    # align image

    (rows, cols) = image.shape[:2]
    l = np.asarray([face_landmark["left_eye"]], dtype=np.int32)
    cl = np.mean(l, axis=1, dtype=np.int32)
    r = np.asarray([face_landmark["right_eye"]], dtype=np.int32)
    cr = np.mean(r, axis=1, dtype=np.int32)
    c = np.mean([cl, cr], axis=0, dtype=np.int32)

    angle = angle_between([1, 0], (cr - cl)[0])
    scale = 100 / np.linalg.norm(cr - cl)

    M = cv2.getRotationMatrix2D((cols / 2, rows / 2), -np.degrees(angle), 1)

    dst = image.copy()
    dst = cv2.warpAffine(dst, M, (cols, rows))
    dst = cv2.resize(dst, None, fx=scale, fy=scale)

    # Remap the center point
    ones = np.ones(shape=(len(c), 1))
    points_ones = np.hstack([c, ones])
    mapped_c = M.dot(scale * points_ones.T).T.astype(int)

    new_top = mapped_c[0][1] - 84
    new_bottom = mapped_c[0][1] + 140
    new_left = mapped_c[0][0] - 112
    new_right = mapped_c[0][0] + 112
    aligned_image = dst[new_top:new_bottom, new_left:new_right]
    return aligned_image


def similarityTransform(inPoints, outPoints):
    s60 = math.sin(60 * math.pi / 180)
    c60 = math.cos(60 * math.pi / 180)

    inPts = np.copy(inPoints).tolist()
    outPts = np.copy(outPoints).tolist()

    xin = (
        c60 * (inPts[0][0] - inPts[1][0])
        - s60 * (inPts[0][1] - inPts[1][1])
        + inPts[1][0]
    )
    yin = (
        s60 * (inPts[0][0] - inPts[1][0])
        + c60 * (inPts[0][1] - inPts[1][1])
        + inPts[1][1]
    )

    inPts.append([np.int(xin), np.int(yin)])

    xout = (
        c60 * (outPts[0][0] - outPts[1][0])
        - s60 * (outPts[0][1] - outPts[1][1])
        + outPts[1][0]
    )
    yout = (
        s60 * (outPts[0][0] - outPts[1][0])
        + c60 * (outPts[0][1] - outPts[1][1])
        + outPts[1][1]
    )

    outPts.append([np.int(xout), np.int(yout)])

    tform = cv2.estimateAffine2D(np.array([inPts]), np.array([outPts]))

    return tform[0]


def tilt_crop_scale(pt_list, image):
    w = 224
    h = 224

    # Eye corners
    eyecornerDst = [
        (np.int(0.25 * w), np.int(h / 3)),
        (np.int(0.75 * w), np.int(h / 3)),
    ]

    pt_list2 = []
    img_list = []
    for points in pt_list:
        dst = image.copy()

        # capture eye information to scaling
        l_eye = points[36]
        r_eye = points[45]
        eyecornerSrc = [l_eye, r_eye]

        # Compute similarity transform
        tform = similarityTransform(eyecornerSrc, eyecornerDst)

        # Apply similarity transformation
        dst = cv2.warpAffine(dst, tform, (w, h))

        # Apply similarity transform on points
        points2 = np.reshape(np.array(points), (72, 1, 2))
        points = cv2.transform(points2, tform)
        points = np.float32(np.reshape(points, (72, 2)))

        pt_list2.append(points)
        img_list.append(dst)
    return img_list, pt_list2


# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2):

    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    t2RectInt = []

    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1] : r1[1] + r1[3], r1[0] : r1[0] + r1[2]]

    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)

    img2Rect = img2Rect * mask

    # Copy triangular region of the rectangular patch to the output image
    img2[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]] = img2[
        r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]
    ] * ((1.0, 1.0, 1.0) - mask)

    img2[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]] = (
        img2[r2[1] : r2[1] + r2[3], r2[0] : r2[0] + r2[2]] + img2Rect
    )


def applyAffineTransform(src, srcTri, dstTri, size):

    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(
        src,
        warpMat,
        (size[0], size[1]),
        None,
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )

    return dst


def constrainPoint(p, w, h):
    p = (min(max(p[0], 0), w - 1), min(max(p[1], 0), h - 1))
    return p


def warp_face(path=None, face_dict=None, average_landmarks=None):
    ROI = cv2.imread(face_dict["ROI"])
    landmarks = np.load(face_dict["LANDMARKS"])

    # Output image
    h = w = 224
    output = np.zeros((h, w, 3), np.float32())
    img = np.zeros((h, w, 3), np.float32())

    # Warp input images to average image landmarks
    dt = Delaunay(average_landmarks).simplices
    # Transform triangles one by ones
    for j in range(0, len(dt)):
        tin = []
        tout = []

        for k in range(0, 3):
            pIn = landmarks[dt[j][k]]
            pIn = constrainPoint(pIn, w, h)

            pOut = average_landmarks[dt[j][k]]
            pOut = constrainPoint(pOut, w, h)

            tin.append(pIn)
            tout.append(pOut)

        warpTriangle(ROI, img, tin, tout)

    # Save image to corresponding folder
    if path != None:
        cv2.imwrite(path, img)
    else:
        return img
