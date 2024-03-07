# FALCON Benchmark and Challenge

This package contains core code for submitting decoders to the FALCON challenge. Full github contains additional examples and documentation.

## Installation
Install `falcon_challenge` with:

```bash
pip install falcon-challenge
```

To create Docker containers for submission, you must have Docker installed.
See, e.g. https://docs.docker.com/desktop/install/linux-install/. Try building and locally testing the provided `sklearn_sample.Dockerfile`, to confirm your setup works. Do this with the following commands (once Docker is installed)

```bash
# Build
sudo docker build -t sk_smoke -f ./sklearn_sample.Dockerfile .
sudo docker run -v ~/projects/stability-benchmark/data:/evaluation_data -it sk_smoke
```



## Submission
To submit to the FALCON benchmark, prepare a decoder and Dockerfile.
- DOCKER instructions todo.

To run local evaluation, first setup a data directory at `./data`.
You can then run:
```bash
python <my_decoder>.py --evaluation local
```