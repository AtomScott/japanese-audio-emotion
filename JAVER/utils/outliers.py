from PIL import Image
from pyod.models.auto_encoder import AutoEncoder

from JAVER.detect_tools.detect import detect_faces
from JAVER.track_tools.track_tools import Face


def remove_outlier_faces(image_paths:list, image_size:int=160) -> list:

    faces = []
    for image_path, bboxes in zip(image_paths, detect_faces(image_paths)):
        im = Image.open(image_path)
        for bbox in bboxes:
            face = Face(idx=image_path, img=im, bbox=bbox)
            faces.append(face)

    clf = AutoEncoder(verbose=1)
    clf.fit([face.embedding for face in faces])
    
    inliers = []
    for face in faces:
        y = clf.predict(embedding.reshape(1,-1))
        
        if y == 0:
            face.face_img.save(image_path)
            inliers.append(embedding)

    logger.info('{:.0%} are outliers'.format(1 - len(inliers) / len(lst)))
    return inliers
