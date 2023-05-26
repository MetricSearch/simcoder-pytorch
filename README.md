# Simcoder
A simple tool for creating embeddings for images for use in similarity experiments.

## Licence
Released under the GNU General Public License v3.0.

Makes use of the SimCLR2 weights from https://github.com/google-research/simclr

Makes use of SimCLR2 weights loader https://github.com/Separius/SimCLRv2-Pytorch

## TODO
[] Remove the SimCLR2 encoder.
[] Add the virtual env to .gitignore.
[] Write make targets that create the virtual env.
[] Write instructions for development and deployment in the README.md.
[] Move the encoding and polyquery into there own subpackages.
[] Find a better way to share non-mutable state between the processes.