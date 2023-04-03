import logging
import time
import click
from encode import encode
from show import show
from list_models import list_models

# let's output the info
logging.basicConfig(level=logging.INFO, format='%(message)')


@click.group(help="CLI tool to encode images for similarity search.")
def cli():
    pass


cli.add_command(encode)
cli.add_command(show)
cli.add_command(list_models)


if __name__ == "__main__":
    start, proc_start = time.time(), time.process_time()
    cli(prog_name="simcoder")
    delta, proc_delta = time.time() - start, time.process_time() - proc_start
    logging.info(f"{delta:.2f}s total, {proc_delta:.2f}s on the CPU.")
