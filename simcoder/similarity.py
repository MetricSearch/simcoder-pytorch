from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torchvision.datasets.folder import default_loader
from scipy.io import loadmat
import torch

# nasty import hack - this is a code smell, work out how to remove it
import sys

sys.path.append("../")
from simcoder.models import get_model

IMAGE_FILES_PER_FOLDER = 10000
mf_dir = Path("/Volumes/Data/mf/images/")  # <<<<<<<<<<<<<<<<<<<<<<<<<< TODO fix me

def get_mf_image(index: int) -> Image.Image:
    folder_idx = index // IMAGE_FILES_PER_FOLDER
    path = mf_dir / str(folder_idx) / f"{index}.jpg"
    img = default_loader(str(path))
    return img


def load_encodings_mat(encodings_dir: Path, key: str = 'features') -> np.array:
    paths = encodings_dir.glob("*.mat")
    paths = sorted(paths, key=lambda p: int(p.stem))
    encodings = [loadmat(p)[key] for p in paths]
    encodings = np.concatenate(encodings)
    return encodings

def load_mf_encodings(encodings_dir: Path) -> np.array:
    return load_encodings_mat(encodings_dir, key='features')

def load_mf_softmax(encodings_dir: Path) -> np.array:
    return load_encodings_mat(encodings_dir, key='probVecs')


def encode(query_image: Image.Image, model_name: str) -> np.array:
    # setup the pytorch device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # get the model
    model, preprocess = get_model(model_name)
    model.to(device)

    # preprocess the image into a tensor, 
    # add a fake batch axis
    # send it to the device (gpu or cpu)
    img = preprocess(query_image).unsqueeze(0).to(device)

    with torch.no_grad():
        features = model(img)
        features = features.detach().cpu().numpy()
    return features

def l1_norm(X):
    X = np.maximum(0,X)
    row_sums = np.sum(X,axis=1)
    X = np.divide(X.T,row_sums).T  # divide all elements rowwise by rowsums!
    return X


def l2_norm(X):
    # This only works if a matrix is passed in fails for vectors of a single row - TODO ho w to fix?
    origin = np.zeros(X.shape[1])
    factor = euclid(origin,X)
    X = np.divide(X.T,factor).T
    return X

def euclid(img_features: np.array, encodings: np.array):
    distances = np.sqrt(np.sum(np.square((img_features - encodings)), axis=1))
    return distances

def getDists(query_index,allData):
    '''Return the distances from the query to allData'''
    '''Returns an array same dimension as allData of scalars'''
    mf_query_data = allData[query_index]
    distances = euclid(mf_query_data, allData)
    return distances

def make_mf_image_grid(
    img_indices: np.array, num_cols: int, num_rows: int, img_w: int, img_h: int
) -> Image.Image:
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
    query_image: Image.Image, encoder_name: str, img_size: int, 
) -> Image.Image:
    features = encode(query_image, encoder_name)
    mf_encodings = load_mf_encodings(Path("/output", f"mf_{encoder_name}"))
    distances = euclid(features, mf_encodings)
    print(np.sort(distances)[:100])
    sorted_indices = np.argsort(distances)
    top_ten_indices = sorted_indices[:100]
    print(top_ten_indices)
    return make_mf_image_grid(top_ten_indices, 10, 10, img_size, img_size)
