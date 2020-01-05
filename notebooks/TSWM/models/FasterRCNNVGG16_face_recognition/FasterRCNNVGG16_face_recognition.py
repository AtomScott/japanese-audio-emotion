import urllib.request
import os
import chainercv

MODEL_URL = 'http://nixeneko.2-d.jp/hatenablog/20170724_facedetection_model/snapshot_model_20180404.npz'
def setup(TRAINED_MODEL=''):
    if not os.path.exists(TRAINED_MODEL):
        download_model(MODEL_URL, TRAINED_MODEL)

    model = chainercv.links.FasterRCNNVGG16(
        n_fg_class=1,
        pretrained_model=TRAINED_MODEL)

    return model


def download_model(url, dest):
    destdir = os.path.dirname(dest)
    if not os.path.exists(destdir):
        os.makedirs(destdir)

    print("Downloading {}... \nThis may take several minutes.".format(dest))
    urllib.request.urlretrieve(url, dest)
