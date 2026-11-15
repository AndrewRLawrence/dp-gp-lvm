"""Readers for the data sets in the ``data`` directory. See ``data/README.md`` for the sources."""

from __future__ import annotations

import csv
import json
import os
from itertools import combinations
from pathlib import Path

import numpy as np

from experiments._common import REPO_ROOT

#: The data directory. Set the ``DP_GP_LVM_DATA`` environment variable to use a different directory.
DATA_DIR = Path(os.environ.get("DP_GP_LVM_DATA", REPO_ROOT / "data"))

#: The labels of the seven HAM10000 diagnostic categories in the skin cancer MNIST data.
SKIN_CANCER_LABELS = tuple(range(7))

_SKIN_CANCER_PREFIX = {2: "two", 3: "two", 6: "six", 7: "seven"}  # The legacy files use "two" for three labels.


def _read_csv(path: Path) -> list[list[str]]:
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    return rows[1:]  # The first row has the column names.


def cmu_swapped_legs() -> np.ndarray:
    """Return CMU subject 35, trial 1: the normal walk and the walk with swapped legs, side by side.

    Returns
    -------
    numpy.ndarray
        The motion, with the shape ``(358, 186)``. Columns 0-92 are the normal walk. Columns 93-185 are the walk with
        the legs swapped.
    """
    with np.load(DATA_DIR / "cmu_mocap" / "35_01.npz") as data:
        return np.array(data["concatenated_motion_cols"], dtype=np.float64)


def cmu_subject7(sequence: int) -> np.ndarray:
    """Return the joint angles of one walking sequence of CMU subject 7.

    Parameters
    ----------
    sequence : int
        The sequence number, from 1 to 12.

    Returns
    -------
    numpy.ndarray
        The joint angles, with the shape ``(frames, 62)``.
    """
    return np.load(DATA_DIR / "cmu_mocap" / f"07_{sequence:02d}_joint_angles.npy")


def cmu_subject7_joint_dims() -> dict[str, int]:
    """Return the number of angle dimensions of each joint of CMU subject 7, in column order."""
    return json.loads((DATA_DIR / "cmu_mocap" / "07_joint_dims.json").read_text())


def frey_faces() -> np.ndarray:
    """Return the Frey faces, scaled to [0, 1].

    Returns
    -------
    numpy.ndarray
        The faces, with the shape ``(1965, 560)``. Each row is a 28 x 20 image.
    """
    from scipy.io import loadmat  # scipy is only needed for this data set.

    faces = loadmat(DATA_DIR / "frey_faces" / "frey_rawface.mat")["ff"]
    return np.asarray(faces, dtype=np.float64).T / 255.0


def colposcopy(modality: str) -> tuple[np.ndarray, np.ndarray]:
    """Return one modality of the UCI Quality Assessment of Digital Colposcopies data.

    Parameters
    ----------
    modality : {"hinselmann", "green", "schiller"}
        The modality.

    Returns
    -------
    tuple of numpy.ndarray
        The 62 predictive attributes and the 7 expert assessments. The last assessment is the consensus.
    """
    rows = np.array(_read_csv(DATA_DIR / "colposcopy" / f"{modality}.csv"), dtype=np.float64)
    return rows[:, :-7], rows[:, -7:]


def skin_cancer_images() -> tuple[np.ndarray, np.ndarray]:
    """Return the 8 x 8 luminance images of the skin cancer MNIST (HAM10000) data, scaled to [0, 1].

    Returns
    -------
    tuple of numpy.ndarray
        The images, with the shape ``(10015, 64)``, and the integer labels.
    """
    rows = _read_csv(DATA_DIR / "skin_cancer_mnist" / "hmnist_8_8_L.csv")
    pixels = np.array([row[:-1] for row in rows], dtype=np.float64) / 255.0
    labels = np.array([int(row[-1]) for row in rows], dtype=int)
    return pixels, labels


def skin_cancer_conditions(labels: tuple[int, ...], num_samples: int = 100) -> np.ndarray:
    """Return one image of each condition in each row: the data for the multi-view skin cancer experiment.

    The function uses the legacy cache file if it exists. It builds the same data with the legacy random stream if
    the file does not exist.

    Parameters
    ----------
    labels : tuple of int
        The condition labels, in increasing order.
    num_samples : int, default 100
        The maximum number of rows.

    Returns
    -------
    numpy.ndarray
        The images, with the shape ``(N, 64 * len(labels))``.
    """
    suffix = "_".join(str(label) for label in labels)
    prefix = _SKIN_CANCER_PREFIX.get(len(labels), str(len(labels)))
    cache = DATA_DIR / "skin_cancer_mnist" / f"{prefix}_conditions_data_{suffix}.npy"
    if cache.exists():
        return np.load(cache)
    images, image_labels = skin_cancer_images()
    groups = [images[image_labels == label] for label in labels]
    num_samples = min(num_samples, *(g.shape[0] for g in groups))
    rng = np.random.RandomState(1)  # The legacy random stream.
    return np.hstack([g[rng.permutation(g.shape[0])[:num_samples]] for g in groups])


def skin_cancer_label_sets(size: int) -> list[tuple[int, ...]]:
    """Return every combination of ``size`` condition labels, as in the legacy experiments."""
    return list(combinations(SKIN_CANCER_LABELS, size))
