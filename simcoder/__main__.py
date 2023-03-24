import logging
from pathlib import Path

import click
import numpy as np
from scipy.io import savemat
from tqdm import tqdm
from pprint import pprint
import torch
from torch.utils.data import DataLoader

from loaders import UnlabelledImageFolder
from models import load_model

logging.basicConfig(level=logging.INFO)


def save_features(arr: np.array, path: Path, format: str) -> None:
    path = path

    match format:
        case "cvs":
            np.savetxt(path, arr, delimiter=",")
        case "npy":
            np.save(path, arr)
        case "mat":
            savemat(path, {"features": arr, "label": "embeddings"})


def encode_images(model, preprocess, input_dir: Path, batch_size: int) -> np.array:
    dataset = UnlabelledImageFolder(input_dir, preprocess)
    loader = DataLoader(dataset, batch_size, num_workers=0)
    features = [model(xs) for xs in tqdm(loader)]
    return torch.cat(features).numpy()


@click.command()
@click.argument("input_dir", type=click.Path(exists=False))
@click.argument("output_path", type=click.Path(exists=False))
@click.argument("model", type=click.STRING)
@click.argument("batch_size", type=click.STRING, default=64)
@click.option("--dirs", "-d", is_flag=True, help="Expect a directory of directories.")
@click.option(
    "--format",
    type=click.Choice(["csv", "npy", "mat"]),
    default="csv",
    help="output format",
)
def encode(input_dir, output_path, model_name, batch_size, dirs, format):
    logging.info("Welcome to Simcoder.")

    # find all the directories to look for images in
    if dirs:
        image_dirs = [f for f in Path(input_dir).iterdir() if f.is_dir()]
        image_dirs = sorted(image_dirs, key=lambda p: int(p.name))
    else:
        image_dirs = input_dir
    logging.info(f"Found {len(image_dirs)} image directories.")

    # get the model from torchvision
    logging.info(f"Loading {model_name} model.")
    model, preprocess = load_model(model_name)

    # iterate over the input dirs, encoding and outputting to disk
    for image_dir in image_dirs:
        logging.info(f"Encoding {input_dir}")
        features = encode_images(model, preprocess, input_dir, batch_size)
        filepath = (output_path / image_dir.stem).with_suffix(f".{format}")
        logging.info(f"Saving embeddings to {filepath}.")
        save_features(features, filepath, format)

    logging.info("Complete.")


if __name__ == "__main__":
    encode()
