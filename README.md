# FALCON Benchmark and Challenge

This package contains core code for submitting decoders to the FALCON challenge. Full github contains additional examples and documentation.

## Installation
Install `falcon_challenge` with:

```bash
pip install falcon-challenge
```

To create Docker containers for submission, you must have Docker installed.
See, e.g. https://docs.docker.com/desktop/install/linux-install/. 

## Getting started

### Data downloading
The FALCON datasets are available on DANDI (or through private correspondence, if beta-testing). 

NOTE FOR BETA TESTERS:
- Some of the sample code expects your data directory to be set up in `./data`. Specifically, the following hierarchy is expected:

`data`
- `h1`
    - `held_in_calib`
    - `held_out_calib`
    - `minival`
    - `eval` (Note this is private data)
- `m1`
    - `sub-MonkeyL-held-in-calib`
    - `sub-MonkeyL-held-out-calib`
    - `minival` (Copy dandiset minival folder into this folder)
    - `eval` (Copy the ground truth held in and held out data into this folder)

H1 should unfold correctly just from unzipping the provided directory. M1 should work by renaming the provided dandiset to `m1` and `minival` folder inside, and then copying the provided eval data into this folder. Each of the lowest level dirs holds the NWB files.

### Code
This codebase contains starter code for implementing your own method for the FALCON challenge. 
- The `falcon_challenge` folder contains the logic for the evaluator. Submitted solutions must conform to the interface specified in `falcon_challenge.interface`.
- In `data_demos`, we provide notebooks that survey each dataset released as part of this challenge.
- In `decoder_demos`, we provide sample decoders and baselines that are formatted to be ready for submission to the challenge. To use them, see the comments in the header of each file ending in `_sample.py`. Your solutions should look similar once implemented!

For example, you can prepare and evaluate a linear decoder by running:
```bash
python decoder_demos/sklearn_decoder.py --training_dir data/h1/held_in_calib/ --calibration_dir data/h1/held_out_calib/ --mode all --task h1
python sklearn_sample.py --evaluation local --phase minival --split h1
```

### Docker Submission
To interface with our challenge, your code will need to be packaged in a Docker container that is submitted to EvalAI. Try this process by building and running the provided `sklearn_sample.Dockerfile`, to confirm your setup works. Do this with the following commands (once Docker is installed)
```bash
# Build
sudo docker build -t sk_smoke -f ./decoder_demos/sklearn_sample.Dockerfile .
sudo docker run -v PATH_TO_YOUR_DATA_DIR:/dataset/evaluation_data -it sk_smoke
## If your solution needs GPUs, append a --gpus all flag to `docker run`
```

## EvalAI Submission
Please ensure that your submission runs locally before running remote evaluation. You can run the previously listed commands, or run `test_docker_local.sh`.
To submit to the FALCON benchmark once your decoder Docker container is ready, follow the instructions on the [EvalAI submission tab](https://eval.ai/web/challenges/challenge-page/2264/submission). It should look something like:
`
evalai push decoder_container:latest --phase <phase-name>
`

### Troubleshooting
Docker:
- If this is your first time with docker, note that `sudo` access is needed, or your user needs to be in the `docker` group. `docker info` should run without error.
- While `sudo` is sufficient for local development, the EvalAI submission step will ultimately require your user to be able to run `docker` commands without `sudo`.
- To do this, [add yourself to the `docker` group](https://docs.docker.com/engine/install/linux-postinstall/). Note you may [need vigr](https://askubuntu.com/questions/964040/usermod-says-account-doesnt-exist-but-adduser-says-it-does) to add your own user.

EvalAI:
- `pip install evalai` may fail on python 3.11, see: https://github.com/aio-libs/aiohttp/issues/6600. We recommend creating a separate env for submission in this case. 
