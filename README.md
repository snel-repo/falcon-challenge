# FALCON Benchmark and Challenge

This repo contains examples that overview FALCON datasets and the `falcon_challenge` package that is needed to submit to the FALCON benchmark.

## Installation
Install `falcon_challenge` with:

```bash
pip install .
```

To create Docker containers for submission, you must have Docker installed.
See, e.g. https://docs.docker.com/desktop/install/linux-install/. Try building and locally testing the provided `sklearn_sample.Dockerfile`, to confirm your setup works.

## Submission
To submit to the FALCON benchmark, prepare a decoder and Dockerfile.
- DOCKER instructions todo.

To run local evaluation, first setup a data directory at `./data`.
You can then run:
```bash
python <my_decoder>.py --evaluation local
```