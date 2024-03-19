# FALCON Benchmark and Challenge

This package contains core code for submitting decoders to the FALCON challenge. Full github contains additional examples and documentation.

## Installation
Install `falcon_challenge` with:

TODO suggest python 3.10

```bash
pip install falcon-challenge
```

To create Docker containers for submission, you must have Docker installed.
See, e.g. https://docs.docker.com/desktop/install/linux-install/. Try building and locally testing the provided `sklearn_sample.Dockerfile`, to confirm your setup works. Do this with the following commands (once Docker is installed)

TODO explanation of training model from sample before running docker smoke test
```bash
# Build
sudo docker build -t sk_smoke -f ./decoder_demos/sklearn_sample.Dockerfile .
sudo docker run -v ~/projects/stability-benchmark/data:/evaluation_data -it sk_smoke
```
Note that additional steps will be needed to allow the docker container to see GPU resources. See [NVIDIA's documentation](https://github.com/NVIDIA/nvidia-container-toolkit) for more information. (The final docker run needs a `--gpus all` flag.)


## Submission
To submit to the FALCON benchmark, prepare a decoder and Dockerfile. The decoder will likely reference source code that must be made importable to the Dockerfile.

To run local evaluation, first setup a data directory at `./data`.

You can then run:
```bash
python <my_decoder>.py --evaluation local --phase <dataset>
```

TODO EvalAI submission instructions