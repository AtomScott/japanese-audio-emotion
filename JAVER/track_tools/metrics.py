import numpy as np
from scipy.spatial.distance import mahalanobis
import itertools as it

from JAVER.track_tools.track_tools import Face, Track

from JAVER.utils.logger import create_logger

logger = create_logger(level='DEBUG')


def _d_1(face, track):
    """Motion Descriptor"""
    assert type(face) == Face
    assert type(track) == Track

    z = track.format_measurement(face)  # new measurement vector

    y, S = track.predict_state()  # should be next (x,y) predict by kalman

    # normalize each array
    z = z / np.linalg.norm(z)
    y = y / np.linalg.norm(y)
    S = S / np.linalg.norm(S)

    dist = mahalanobis(z, y, S)
    return dist


def _d_2(face, track):
    """Appearance Descriptor"""
    assert type(face) == Face
    assert type(track) == Track

    dist = 1
    r_j = face.embedding
    for _face in track.gallery:
        r_i = _face.embedding
        r_j_dist = r_j.T @ r_i
        # assert 0 <= r_j_dist <= 1
        if not 0 <= r_j_dist <= 1: logger.warning(r_j_dist)
        dist = min(dist, r_j_dist)
    return 1 - dist


def association_cost_matrix(faces, tracks, lam=0.1):
    n_faces = len(faces)
    n_tracks = len(tracks)
    C = np.zeros((n_faces, n_tracks))
    for i, j in it.product(range(n_faces), range(n_tracks)):
        C[i, j] = lam * _d_1(faces[i], tracks[j]) + (1 - lam) * _d_2(faces[i], tracks[j])
    return C


def gate_matrix(faces, tracks, _thresh_1=50, _thresh_2=1):
    n_faces = len(faces)
    n_tracks = len(tracks)

    G = np.zeros((n_faces, n_tracks))
    for i, j in it.product(range(n_faces), range(n_tracks)):
        G[i, j] = (_d_1(faces[i], tracks[j]) <= _thresh_1) * (_d_2(faces[i], tracks[j]) <= _thresh_2)
    G = G.astype(int)

    return G
