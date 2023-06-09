import logging
import time
import click
import multiprocessing as mp
import sys

from sisap2023.encoders.encode import encode
from sisap2023.encoders.show import show
from sisap2023.encoders.list_models import list_models
from sisap2023.experiments.experiment import experiment
from sisap2023.experiments.experiment_strict import experimentstrict
from sisap2023.experiments.experimentselected import experimentselected


# let's output the info
logging.basicConfig(level=logging.INFO, format='%(message)')


@click.group(help="CLI tool to encode images for similarity search.")
def cli():
    pass


cli.add_command(encode)
cli.add_command(show)
cli.add_command(list_models)
cli.add_command(experiment)
cli.add_command(experimentstrict)
cli.add_command(experimentselected)

if __name__ == "__main__":
    mp.set_start_method("fork")
    start, proc_start = time.time(), time.process_time()
    cli(prog_name="simcoder")
    delta, proc_delta = time.time() - start, time.process_time() - proc_start
    logging.info(f"{delta:.2f}s total, {proc_delta:.2f}s on the CPU.")
