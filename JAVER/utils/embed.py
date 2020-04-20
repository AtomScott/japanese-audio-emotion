def embed_faces(in_image_paths, save_embeddings=True, image_size=160, replace_images=False):
    """Crops face(faces) in image and return the cropped area(areas) along with an embedding(embeddings).

    
    
    Parameters
    ----------
    in_image_paths : list
        Path to images to crop
    save_embeddings : bool, optional
        Save the embeddings, by default True
    image_size : int, optional
        [description], by default 160
    replace_images : bool, optional
        [description], by default False
    
    Returns
    -------
    [type]
        [description]
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
