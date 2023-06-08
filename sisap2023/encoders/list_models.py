import logging

import click

from models import get_availible_models


@click.command()
def list_models():
    logging.info("Welcome to Simcoder.")
    model_names = get_availible_models()
    for name in model_names:
        print(name)
