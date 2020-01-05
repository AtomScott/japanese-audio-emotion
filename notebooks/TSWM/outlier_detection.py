from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.utils import shuffle
from sklearn.utils.random import sample_without_replacement
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import warnings
import matplotlib.pyplot as plt
import os

# なるべくどんなデータでも対応できるようにしたいね
# ひとつのパラメータでみたいな、、
# 　pythonでありそうやなライブラリが
# とりあえず、SVMベースのものを使うことにする
def purify(path_dict, name_list):
    warnings.simplefilter("ignore")
    del_list = []
    for line in name_list:
        name, yt_url = line.split(" ")

        one_dict = {}
        for key, face_dict in path_dict.items():
            if key.split("_")[0] == name:
                one_dict[key] = face_dict

        ox = np.asarray(
            [np.load(face_dict["EMBEDDING"])[0] for _, face_dict in one_dict.items()]
        )

        clf = OneClassSVM()
        model = clf.fit(ox)
        score = 0
        for key, face_dict in one_dict.items():
            x = np.load(face_dict["EMBEDDING"])[0].reshape(1, -1)
            if model.predict(x)[0] == -1:
                del_list.append(key)
                score += 1
        print(
            "[INFO] Removed {0:0.3f}% Images for class {1}".format(
                score / len(one_dict), name
            )
        )

    for key in del_list:
        os.remove(path_dict[key]["ROI"])
        os.remove(path_dict[key]["LANDMARKS"])
        os.remove(path_dict[key]["EMBEDDING"])
        del path_dict[key]
    return path_dict
