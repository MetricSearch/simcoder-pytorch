from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.io import loadmat

# nasty import hack - this is a code smell, work out how to remove it
import sys

sys.path.append("../")
from simcoder.models import get_model


IMAGE_FILES_PER_FOLDER = 10000
mf_dir = Path("/input/")


def get_mf_image(index: int) -> Image.Image:
    folder_idx = index // IMAGE_FILES_PER_FOLDER
    path = mf_dir / str(folder_idx) / f"{index}.jpg"
    img = Image.open(path)  # TODO: work out how to close this file pointer
    return img


def load_mf_encodings(encodings_dir: Path) -> np.array:
    paths = encodings_dir.glob("*.mat")
    paths = sorted(paths, key=lambda p: int(p.stem))
    encodings = [loadmat(p)["features"] for p in paths]
    encodings = np.concatenate(encodings)
    return encodings


def encode(query_image: Image.Image, model_name: str) -> np.array:
    model, preprocess = get_model(model_name)
    img = preprocess(query_image)
    features = model(img.unsqueeze(0))
    features = features.squeeze().detach().cpu().numpy()
    return features


def euclid(img_features: np.array, encodings: np.array):
    distances = np.sqrt(np.sum((img_features - encodings) ** 2, axis=1))
    return distances


def make_mf_image_grid(
    img_indices: np.array, num_cols: int, num_rows: int, img_w: int, img_h: int
):
    images = [get_mf_image(i) for i in img_indices]
    grid = Image.new("RGB", size=(num_cols * img_w, num_rows * img_h))
    for idx, img in enumerate(images):
        img = img.resize((img_w, img_h), Image.Resampling.BILINEAR)
        grid.paste(img, box=(idx % num_cols * img_w, idx // num_cols * img_h))
    return grid


def show_features(features: np.array):
    plt.plot(features)
    plt.show()


def get_similar_mf(
    query_image: Image.Image, encoder_name: str, img_size: int
) -> Image.Image:
    features = encode(query_image, encoder_name)
    mf_encodings = load_mf_encodings(Path("/output", f"mf_{encoder_name}"))
    distances = euclid(features, mf_encodings)
    sorted_indices = np.argsort(distances)
    return make_mf_image_grid(sorted_indices[:100], 10, 10, img_size, img_size)
