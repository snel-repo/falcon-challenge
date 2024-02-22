# FALCON Benchmark and Challenge

This repo contains examples that overview FALCON datasets and the `falcon_challenge` package that is needed to submit to the FALCON benchmark.

## Installation
Install `falcon_challenge` with:

```bash
pip install .
```

## Submission
To submit to the FALCON benchmark, prepare a decoder and Dockerfile.
- DOCKER instructions todo.

To run local evaluation, first setup a data directory at `./data`.
You can then run:
```bash
python <my_decoder>.py --evaluation local
```