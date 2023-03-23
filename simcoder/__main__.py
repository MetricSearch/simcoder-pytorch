import logging
from pathlib import Path

import click
import numpy as np
from scipy.io import savemat
from tqdm import tqdm
from pprint import pprint

import os


logging.basicConfig(level=logging.INFO)


def load_model(model_name: str, weights: str, layer: str):
    exec(f"from torchvision.models import {model_name}")
    model = eval(f"{model_name}(weights='{weights}')")
    return model


def save_features(arr: np.array, path: Path, format: str) -> None:
    logging.info(f"Saving embeddings to {path}.")
    path = path.with_suffix(f".{format}")

    match format:
        case "cvs":
            np.savetxt(path, arr, delimiter=",")
        case "npy":
            np.save(path, arr)
        case "mat":
            savemat(path, {"features": arr, "label": "embeddings"})


def encode_images_in_dir(model, input_dir: Path, batch_size: int) -> np.array:
    # load in the image from the input_dir
    logging.info(f"Loading image dataset from {input_dir}")

    return features


@click.command()
@click.argument("input_dir", type=click.Path(exists=False))
@click.argument("output_path", type=click.Path(exists=False))
@click.argument("model", type=click.STRING)
@click.argument("weights", type=click.STRING)
@click.argument("layer", type=click.STRING)
@click.argument("batch_size", type=click.STRING, default=64)
@click.option("--dirs", "-d", is_flag=True, help="Expect a directory of directories.")
@click.option(
    "--format",
    type=click.Choice(["csv", "npy", "mat"]),
    default="csv",
    help="output format",
)
@click.option(
    "--chunksize",
    type=int,
    help="Number of embeddings in each output file.",
)
def encode(
    input_dir,
    output_path,
    model_name,
    weights,
    layer,
    batch_size,
    dirs,
    format,
    chunksize,
):
    logging.info("Welcome to Simcoder.")

    # find all the directories to look for images in
    if dirs:
        image_dirs = [f for f in Path(input_dir).iterdir() if f.is_dir()]
        image_dirs = sorted(image_dirs, key=lambda p: int(p.name))
    else:
        image_dirs = input_dir
    logging.info(f"Found {len(image_dirs)} image directories.")

    # get the model from torchvision
    model = load_model(model_name, weights, layer)

    # iterate over the input dirs, encoding and outputting to disk
    for image_dir in image_dirs:
        features = encode_images_in_dir(model, input_dir, batch_size)
        filepath = output_path / image_dir.stem
        save_features(features, filepath, format)

    logging.info(f"Complete.")


if __name__ == "__main__":
    encode()
