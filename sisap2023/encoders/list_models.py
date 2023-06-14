import logging

import click

from sisap2023.encoders.models import get_availible_models


@click.command()
def list_models():
    logging.info("Welcome to sisap2023.")
    model_names = get_availible_models()
    for name in model_names:
        print(name)
