from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.datasets.folder import default_loader
from scipy.io import loadmat
import torch

from sisap2023.encoders.models import get_model

IMAGE_FILES_PER_FOLDER = 10000
mf_dir = None

def set_mf_images_path(path: Path) -> None:
    global mf_dir
    assert path.exists(), "Path to mf images does not exist."
    mf_dir = path

def get_mf_image(index: int) -> Image.Image:
    assert mf_dir, "mf_dir is not set. Please set it to a Path to the mf images."
    folder_idx = index // IMAGE_FILES_PER_FOLDER
    path = mf_dir / str(folder_idx) / f"{index}.jpg"
    img = default_loader(str(path))
    return img

def make_mf_image_grid(
    img_indices: np.array, num_cols: int, num_rows: int, img_w: int, img_h: int
) -> Image.Image:
    assert mf_dir, "mf_dir is not set. Please set it to a Path to the mf images."    
    images = [get_mf_image(i) for i in img_indices]
    grid = Image.new("RGB", size=(num_cols * img_w, num_rows * img_h))
    for idx, img in enumerate(images):
        img = img.resize((img_w, img_h), Image.Resampling.BILINEAR)
        grid.paste(img, box=(idx % num_cols * img_w, idx // num_cols * img_h))
    return grid

def load_encodings(encodings_dir: Path, key: str = 'features') -> np.array:
    paths = encodings_dir.glob("*.mat")
    paths = sorted(paths, key=lambda p: int(p.stem))
    encodings = [loadmat(p)[key] for p in paths]
    encodings = np.concatenate(encodings)
    return encodings
