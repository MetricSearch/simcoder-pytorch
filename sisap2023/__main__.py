import logging
import time
import click
import multiprocessing as mp

from sisap2023.encoders.encode import encode
from sisap2023.encoders.show import show
from sisap2023.encoders.list_models import list_models
from sisap2023.experiments.experiment import experiment


# let's output the info
logging.basicConfig(level=logging.INFO, format='%(message)')


@click.group(help="CLI tool to encode images for similarity search.")
def cli():
    pass


cli.add_command(encode)
cli.add_command(show)
cli.add_command(list_models)
cli.add_command(experiment)

if __name__ == "__main__":
    mp.set_start_method("fork")
    cli(prog_name="sisap2023")
