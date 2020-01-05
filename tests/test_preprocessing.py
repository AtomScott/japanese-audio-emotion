import os, sys, shutil
# sys.path.append(os.pardir)
from JVAER import utils

# def test_make_save_dirs():
#     path = os.path.join('tests','assets')
#     utils.make_save_dirs(path, 'sample', overwrite=False)
#     assert sorted(os.path.join(path, d) for d in os.listdir(path)) == sorted([os.path.join(path, d) for d in ['images', 'videos', 'embeddings']])

# def test_embed_faces():
#     in_image_paths = [
#         os.path.join('tests','sample_image_1.jpeg'), # image of 3 people
#         os.path.join('tests','sample_image_2.jpeg') # image of 2 people
#         ]

#     embeddings, faces, boxes = utils.embed_faces(in_image_paths, save_embeddings=True, image_size=100, replace_images=False)

#     assert (len(embeddings), embeddings[0].shape, embeddings[1].shape) == (2, (512,) )
#     as>sert len(faces) == 3
#     assert faces[0].shape == (3, 100, 100)
#     assert len(boxes) == 3

def test_crop_faces():
    in_image_paths = [
        os.path.join('tests','sample_image_1.jpeg'), # image of 3 people
        os.path.join('tests','sample_image_2.jpeg') # image of 2 people
        ]

    out_paths = [
        os.path.join('tests','cropped','sample_image_1.jpeg'), # image of 3 people
        os.path.join('tests','cropped','sample_image_2.jpeg') # image of 2 people
    ]

    bboxes, out_paths, rois = utils.crop_faces(in_image_paths, out_paths=out_paths).values()

    assert all(list(map(lambda x: len(x)==5, [bboxes, out_paths, rois])))

def test_embed_faces():
    in_image_paths = [os.path.join('tests','cropped','sample_image_1_1.jpg')]
    out_paths = [os.path.join('tests','embeddings','sample_image_1_1.npy')]
    
    embeddings, out_paths = utils.embed_faces(in_image_paths=in_image_paths, out_paths=out_paths).values()
    
    assert len(embeddings) == len(out_paths)

