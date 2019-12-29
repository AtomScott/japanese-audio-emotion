---
layout: post
title: code.get_talking_sequence
description: >
  
---

# code.get_talking_sequence
---
##### Parameters {#section}

<dl>
<dt markdown='1'>`hidden_neurons` : *list, optional (default=[64, 32, 32, 64])*
</dt>
	<dd markdown='1'> The number of neurons per hidden layers. 
</dd>

<dt markdown='1'>`hidden_activation` : *str, optional (default='relu')*
</dt>
	<dd markdown='1'> Activation function to use for hidden layers.All hidden layers are forced to use the same type of activation.See https://keras.io/activations/ 
</dd>

<dt markdown='1'>`output_activation` : *str, optional (default='sigmoid')*
</dt>
	<dd markdown='1'> Activation function to use for output layer.See https://keras.io/activations/ 
</dd>

<dt markdown='1'>`loss` : *str or obj, optional (default=keras.losses.mean_squared_error)*
</dt>
	<dd markdown='1'> String (name of objective function) or objective function.See https://keras.io/losses/ 
</dd>

<dt markdown='1'>`optimizer` : *str, optional (default='adam')*
</dt>
	<dd markdown='1'> String (name of optimizer) or optimizer instance.See https://keras.io/optimizers/ 
</dd>

<dt markdown='1'>`epochs` : *int, optional (default=100)*
</dt>
	<dd markdown='1'> Number of epochs to train the model. 
</dd>

<dt markdown='1'>`batch_size` : *int, optional (default=32)*
</dt>
	<dd markdown='1'> Number of samples per gradient update. 
</dd>

<dt markdown='1'>`dropout_rate` : *float in (0., 1), optional (default=0.2)*
</dt>
	<dd markdown='1'> The dropout to be used across all layers. 
</dd>

<dt markdown='1'>`l2_regularizer` : *float in (0., 1), optional (default=0.1)*
</dt>
	<dd markdown='1'> The regularization strength of activity_regularizerapplied on each layer. By default, l2 regularizer is used. Seehttps://keras.io/regularizers/ 
</dd>

<dt markdown='1'>`validation_size` : *float in (0., 1), optional (default=0.1)*
</dt>
	<dd markdown='1'> The percentage of data to be used for validation. 
</dd>

<dt markdown='1'>`preprocessing` : *bool, optional (default=True)*
</dt>
	<dd markdown='1'> If True, apply standardization on the data. 
</dd>

<dt markdown='1'>`verbose` : *int, optional (default=1)*
</dt>
	<dd markdown='1'> Verbosity mode.- 0 = silent- 1 = progress bar- 2 = one line per epoch.For verbosity >= 1, model summary may be printed. 
</dd>

<dt markdown='1'>`random_state` : *random_state: int, RandomState instance or None, optional*
</dt>
	<dd markdown='1'> (default=None)If int, random_state is the seed used by the randomnumber generator; If RandomState instance, random_state is the randomnumber generator; If None, the random number generator is theRandomState instance used by `np.random`. 
</dd>

<dt markdown='1'>`contamination` : *float in (0., 0.5), optional (default=0.1)*
</dt>
	<dd markdown='1'> The amount of contamination of the data set, i.e.the proportion of outliers in the data set. When fitting this is usedto define the threshold on the decision function. 
</dd>

</dl>

<div class='desc' markdown="1">
##### Parameters {#section}

<dl>
<dt markdown='1'>`hidden_neurons` : *list, optional (default=[64, 32, 32, 64])*
</dt>
	<dd markdown='1'> The number of neurons per hidden layers. 
</dd>

<dt markdown='1'>`hidden_activation` : *str, optional (default='relu')*
</dt>
	<dd markdown='1'> Activation function to use for hidden layers.All hidden layers are forced to use the same type of activation.See https://keras.io/activations/ 
</dd>

<dt markdown='1'>`output_activation` : *str, optional (default='sigmoid')*
</dt>
	<dd markdown='1'> Activation function to use for output layer.See https://keras.io/activations/ 
</dd>

<dt markdown='1'>`loss` : *str or obj, optional (default=keras.losses.mean_squared_error)*
</dt>
	<dd markdown='1'> String (name of objective function) or objective function.See https://keras.io/losses/ 
</dd>

<dt markdown='1'>`optimizer` : *str, optional (default='adam')*
</dt>
	<dd markdown='1'> String (name of optimizer) or optimizer instance.See https://keras.io/optimizers/ 
</dd>

<dt markdown='1'>`epochs` : *int, optional (default=100)*
</dt>
	<dd markdown='1'> Number of epochs to train the model. 
</dd>

<dt markdown='1'>`batch_size` : *int, optional (default=32)*
</dt>
	<dd markdown='1'> Number of samples per gradient update. 
</dd>

<dt markdown='1'>`dropout_rate` : *float in (0., 1), optional (default=0.2)*
</dt>
	<dd markdown='1'> The dropout to be used across all layers. 
</dd>

<dt markdown='1'>`l2_regularizer` : *float in (0., 1), optional (default=0.1)*
</dt>
	<dd markdown='1'> The regularization strength of activity_regularizerapplied on each layer. By default, l2 regularizer is used. Seehttps://keras.io/regularizers/ 
</dd>

<dt markdown='1'>`validation_size` : *float in (0., 1), optional (default=0.1)*
</dt>
	<dd markdown='1'> The percentage of data to be used for validation. 
</dd>

<dt markdown='1'>`preprocessing` : *bool, optional (default=True)*
</dt>
	<dd markdown='1'> If True, apply standardization on the data. 
</dd>

<dt markdown='1'>`verbose` : *int, optional (default=1)*
</dt>
	<dd markdown='1'> Verbosity mode.- 0 = silent- 1 = progress bar- 2 = one line per epoch.For verbosity >= 1, model summary may be printed. 
</dd>

<dt markdown='1'>`random_state` : *random_state: int, RandomState instance or None, optional*
</dt>
	<dd markdown='1'> (default=None)If int, random_state is the seed used by the randomnumber generator; If RandomState instance, random_state is the randomnumber generator; If None, the random number generator is theRandomState instance used by `np.random`. 
</dd>

<dt markdown='1'>`contamination` : *float in (0., 0.5), optional (default=0.1)*
</dt>
	<dd markdown='1'> The amount of contamination of the data set, i.e.the proportion of outliers in the data set. When fitting this is usedto define the threshold on the decision function. 
</dd>

</dl>

##### Parameters {#section}

<dl>
<dt markdown='1'>`hidden_neurons` : *list, optional (default=[64, 32, 32, 64])*
</dt>
	<dd markdown='1'> The number of neurons per hidden layers. 
</dd>

<dt markdown='1'>`hidden_activation` : *str, optional (default='relu')*
</dt>
	<dd markdown='1'> Activation function to use for hidden layers.All hidden layers are forced to use the same type of activation.See https://keras.io/activations/ 
</dd>

<dt markdown='1'>`output_activation` : *str, optional (default='sigmoid')*
</dt>
	<dd markdown='1'> Activation function to use for output layer.See https://keras.io/activations/ 
</dd>

<dt markdown='1'>`loss` : *str or obj, optional (default=keras.losses.mean_squared_error)*
</dt>
	<dd markdown='1'> String (name of objective function) or objective function.See https://keras.io/losses/ 
</dd>

<dt markdown='1'>`optimizer` : *str, optional (default='adam')*
</dt>
	<dd markdown='1'> String (name of optimizer) or optimizer instance.See https://keras.io/optimizers/ 
</dd>

<dt markdown='1'>`epochs` : *int, optional (default=100)*
</dt>
	<dd markdown='1'> Number of epochs to train the model. 
</dd>

<dt markdown='1'>`batch_size` : *int, optional (default=32)*
</dt>
	<dd markdown='1'> Number of samples per gradient update. 
</dd>

<dt markdown='1'>`dropout_rate` : *float in (0., 1), optional (default=0.2)*
</dt>
	<dd markdown='1'> The dropout to be used across all layers. 
</dd>

<dt markdown='1'>`l2_regularizer` : *float in (0., 1), optional (default=0.1)*
</dt>
	<dd markdown='1'> The regularization strength of activity_regularizerapplied on each layer. By default, l2 regularizer is used. Seehttps://keras.io/regularizers/ 
</dd>

<dt markdown='1'>`validation_size` : *float in (0., 1), optional (default=0.1)*
</dt>
	<dd markdown='1'> The percentage of data to be used for validation. 
</dd>

<dt markdown='1'>`preprocessing` : *bool, optional (default=True)*
</dt>
	<dd markdown='1'> If True, apply standardization on the data. 
</dd>

<dt markdown='1'>`verbose` : *int, optional (default=1)*
</dt>
	<dd markdown='1'> Verbosity mode.- 0 = silent- 1 = progress bar- 2 = one line per epoch.For verbosity >= 1, model summary may be printed. 
</dd>

<dt markdown='1'>`random_state` : *random_state: int, RandomState instance or None, optional*
</dt>
	<dd markdown='1'> (default=None)If int, random_state is the seed used by the randomnumber generator; If RandomState instance, random_state is the randomnumber generator; If None, the random number generator is theRandomState instance used by `np.random`. 
</dd>

<dt markdown='1'>`contamination` : *float in (0., 0.5), optional (default=0.1)*
</dt>
	<dd markdown='1'> The amount of contamination of the data set, i.e.the proportion of outliers in the data set. When fitting this is usedto define the threshold on the decision function. 
</dd>

</dl>

Auto Encoder (AE) is a type of neural networks for learning useful data
representations unsupervisedly. Similar to PCA, AE could be used to
detect outlying objects in the data by calculating the reconstruction
errors. See :cite:`aggarwal2015outlier` Chapter 3 for details.
---
</div>Model parameters can be loaded based on pretraining on the VGGFace2 or CASIA-Webface
datasets. Pretrained state_dicts are automatically downloaded on model instantiation if
requested and cached in the torch cache. Subsequent instantiations use the cache rather than
redownloading.

Keyword Arguments:
    pretrained {str} -- Optional pretraining dataset. Either 'vggface2' or 'casia-webface'.
        (default: {None})
    classify {bool} -- Whether the model should output classification probabilities or feature
        embeddings. (default: {False})
    num_classes {int} -- Number of output classes. If 'pretrained' is set and num_classes not
        equal to that used for the pretrained model, the final linear layer will be randomly
        initialized. (default: {None})
    dropout_prob {float} -- Dropout probability. (default: {0.6})
<div class='desc' markdown="1">
Model parameters can be loaded based on pretraining on the VGGFace2 or CASIA-Webface
datasets. Pretrained state_dicts are automatically downloaded on model instantiation if
requested and cached in the torch cache. Subsequent instantiations use the cache rather than
redownloading.

Keyword Arguments:
    pretrained {str} -- Optional pretraining dataset. Either 'vggface2' or 'casia-webface'.
        (default: {None})
    classify {bool} -- Whether the model should output classification probabilities or feature
        embeddings. (default: {False})
    num_classes {int} -- Number of output classes. If 'pretrained' is set and num_classes not
        equal to that used for the pretrained model, the final linear layer will be randomly
        initialized. (default: {None})
    dropout_prob {float} -- Dropout probability. (default: {0.6})
Inception Resnet V1 model with optional loading of pretrained weights.
---
</div>This class loads pretrained P-, R-, and O-nets and, given raw input images as PIL images,
returns images cropped to include the face only. Cropped faces can optionally be saved to file
also.

Keyword Arguments:
    image_size {int} -- Output image size in pixels. The image will be square. (default: {160})
    margin {int} -- Margin to add to bounding box, in terms of pixels in the final image. 
        Note that the application of the margin differs slightly from the davidsandberg/facenet
        repo, which applies the margin to the original image before resizing, making the margin
        dependent on the original image size (this is a bug in davidsandberg/facenet).
        (default: {0})
    min_face_size {int} -- Minimum face size to search for. (default: {20})
    thresholds {list} -- MTCNN face detection thresholds (default: {[0.6, 0.7, 0.7]})
    factor {float} -- Factor used to create a scaling pyramid of face sizes. (default: {0.709})
    prewhiten {bool} -- Whether or not to prewhiten images before returning. (default: {True})
    select_largest {bool} -- If True, if multiple faces are detected, the largest is returned.
        If False, the face with the highest detection probability is returned. (default: {True})
    keep_all {bool} -- If True, all detected faces are returned, in the order dictated by the
        select_largest parameter. If a save_path is specified, the first face is saved to that
        path and the remaining faces are saved to <save_path>1, <save_path>2 etc.
    device {torch.device} -- The device on which to run neural net passes. Image tensors and
        models are copied to this device before running forward passes. (default: {None})
<div class='desc' markdown="1">
This class loads pretrained P-, R-, and O-nets and, given raw input images as PIL images,
returns images cropped to include the face only. Cropped faces can optionally be saved to file
also.

Keyword Arguments:
    image_size {int} -- Output image size in pixels. The image will be square. (default: {160})
    margin {int} -- Margin to add to bounding box, in terms of pixels in the final image. 
        Note that the application of the margin differs slightly from the davidsandberg/facenet
        repo, which applies the margin to the original image before resizing, making the margin
        dependent on the original image size (this is a bug in davidsandberg/facenet).
        (default: {0})
    min_face_size {int} -- Minimum face size to search for. (default: {20})
    thresholds {list} -- MTCNN face detection thresholds (default: {[0.6, 0.7, 0.7]})
    factor {float} -- Factor used to create a scaling pyramid of face sizes. (default: {0.709})
    prewhiten {bool} -- Whether or not to prewhiten images before returning. (default: {True})
    select_largest {bool} -- If True, if multiple faces are detected, the largest is returned.
        If False, the face with the highest detection probability is returned. (default: {True})
    keep_all {bool} -- If True, all detected faces are returned, in the order dictated by the
        select_largest parameter. If a save_path is specified, the first face is saved to that
        path and the remaining faces are saved to <save_path>1, <save_path>2 etc.
    device {torch.device} -- The device on which to run neural net passes. Image tensors and
        models are copied to this device before running forward passes. (default: {None})
MTCNN face detection module.
---
</div>MTCNN face detection module.
<div class='desc' markdown="1">
---
</div>MTCNN face detection module.
<div class='desc' markdown="1">
MTCNN face detection module.
Core developer interface for pytube.
---
</div>These methods will be called by the parser:
  reset()
  feed(markup)

The tree builder may call these methods from its feed() implementation:
  handle_starttag(name, attrs) # See note about return value
  handle_endtag(name)
  handle_data(data) # Appends to the current data node
  endData(containerClass) # Ends the current data node

No matter how complicated the underlying parser is, you should be
able to build a tree using 'start tag' events, 'end tag' events,
'data' events, and "done with data" events.

If you encounter an empty-element tag (aka a self-closing tag,
like HTML's <br> tag), call handle_starttag and then
handle_endtag.
<div class='desc' markdown="1">
These methods will be called by the parser:
  reset()
  feed(markup)

The tree builder may call these methods from its feed() implementation:
  handle_starttag(name, attrs) # See note about return value
  handle_endtag(name)
  handle_data(data) # Appends to the current data node
  endData(containerClass) # Ends the current data node

No matter how complicated the underlying parser is, you should be
able to build a tree using 'start tag' events, 'end tag' events,
'data' events, and "done with data" events.

If you encounter an empty-element tag (aka a self-closing tag,
like HTML's <br> tag), call handle_starttag and then
handle_endtag.
These methods will be called by the parser:
  reset()
  feed(markup)

The tree builder may call these methods from its feed() implementation:
  handle_starttag(name, attrs) # See note about return value
  handle_endtag(name)
  handle_data(data) # Appends to the current data node
  endData(containerClass) # Ends the current data node

No matter how complicated the underlying parser is, you should be
able to build a tree using 'start tag' events, 'end tag' events,
'data' events, and "done with data" events.

If you encounter an empty-element tag (aka a self-closing tag,
like HTML's <br> tag), call handle_starttag and then
handle_endtag.
This class defines the basic interface called by the tree builders.
---
</div>#### **code.logger.create_logger(** *level='INFO', name='code.logger'*  **)** {#signature}

<div class='desc' markdown="1">
---
</div>#### **code.get_talking_sequence.detect_outliers(** *lst*  **)** {#signature}

<div class='desc' markdown="1">
---
</div>Arguments:
    query : str
        Query to download images for.
    n_images : int
        Number of images to download.
    out_dir_path : str
        Output path for images.
    chromedriver_path : str
        Path to chromedriver

Keyword Arguments:
    chromedriver_path {str} -- Path to chromedriver (default: {"./chromedriver"})

Returns:
    out_image_paths {list} -- list of absolute paths to images
<div class='desc' markdown="1">
#### **code.get_talking_sequence.download_images(** *query, n_images, out_dir_path, chromedriver_path='./chromedriver'*  **)** {#signature}

Downloads face images for a query.
---
</div>Arguments:
    image_paths {list} -- list of paths to images

Keyword Arguments:
    save {bool} -- If True, saves the cropped faces and embeddings to a file and returns the out_paths. (default: {False})

Returns:
   embeddings, faces {[ndarray], [ndarray]} -- if save==False
   embedding_paths, face_paths {[str], [str]} -- if save==True
<div class='desc' markdown="1">
#### **code.get_talking_sequence.embed_faces(** *in_image_paths, save_embeddings=True, image_size=160, replace_images=False*  **)** {#signature}

Crops face(faces) in image and return the cropped area(areas) along with an embedding(embeddings).
---
</div>#### **moviepy.video.io.ffmpeg_tools.ffmpeg_extract_subclip(** *filename, t1, t2, targetname=None*  **)** {#signature}

<div class='desc' markdown="1">
Makes a new video file playing video file ``filename`` between
the times ``t1`` and ``t2``. 
---
</div>Returns:
    args {argparse.Namespace} -- dict of arguments
<div class='desc' markdown="1">
#### **code.arguments.parse_args(** **  **)** {#signature}

Custom argument parser for command line usage.
---
</div>#### **facenet_pytorch.models.mtcnn.prewhiten(** *x*  **)** {#signature}

<div class='desc' markdown="1">
---
</div>#### **code.get_talking_sequence.scrape_videos(** *query, out_path, n*  **)** {#signature}

<div class='desc' markdown="1">
---
</div>#### **code.get_talking_sequence.scrape_videos(** *query, out_path, n*  **)** {#signature}

<div class='desc' markdown="1">
#### **code.get_talking_sequence.scrape_videos(** *query, out_path, n*  **)** {#signature}

Decorate an iterable object, returning an iterator which acts exactly
like the original iterable, but prints a dynamically updating
progressbar every time a value is requested.
---
</div>