# FALCON Benchmark and Challenge

This package contains core code for submitting decoders to the FALCON challenge. For a more general overview of FALCON, please see the [main website](https://snel-repo.github.io/falcon/).

## Installation
Install `falcon_challenge` with:

```bash
pip install falcon-challenge
```

To create Docker containers for submission, you must have Docker installed.
See, e.g. https://docs.docker.com/desktop/install/linux-install/. 

## Getting started

### Data downloading
The FALCON datasets are available on DANDI ([H1](https://dandiarchive.org/dandiset/000954?search=falcon&pos=3), [H2](https://dandiarchive.org/dandiset/000950?search=falcon&pos=4), [M1](https://dandiarchive.org/dandiset/000941?search=falcon&pos=1), [M2](https://dandiarchive.org/dandiset/000953?search=falcon&pos=2), [B1](https://dandiarchive.org/dandiset/001046)). H1 and H2 are human intractorical brain-computer interface (iBCI) datasets, M1 and M2 are monkey iBCI datasets, and B1 is a songbird iBCI dataset.

Data from each dataset is broken down as follows:

- Held-in 
    - Data from the first several recording sessions. 
    - All non-evaluation data is released and split into calibration (large portion) and minival (small portion) sets. 
    - Held-in calibration data is intended to train decoders from scratch.
    - Minival data enables validation of held-in decoders and submission debugging.
- Held-out: 
    - Data from the latter several recording sessions. 
    - A small portion of non-evaluation data is released for calibration. 
    - Held-out calibration data is intentionally small to discourage training decoders from scratch on this data and provides an opportunity for few-shot recalibration.

Some of the sample code expects your data directory to be set up in `./data`. Specifically, the following hierarchy is expected:

`data`
- `h1`
    - `held_in_calib`
    - `held_out_calib`
    - `minival` (Copy dandiset minival folder into this folder)
- `h2`
    - `held_in_calib`
    - `held_out_calib`
    - `minival` (Copy dandiset minival folder into this folder)
- `m1`
    - `sub-MonkeyL-held-in-calib`
    - `sub-MonkeyL-held-out-calib`
    - `minival` (Copy dandiset minival folder into this folder)
- `m2`
    - `held_in_calib`
    - `held_out_calib`
    - `minival` (Copy dandiset minival folder into this folder)
<!-- - `b1`
    - `held_in_calib`
    - `held_out_calib`
    - `minival` (Copy dandiset minival folder into this folder) -->

Each of the lowest level dirs holds the data files (in Neurodata Without Borders (NWB) format). Data from some sessions is distributed across multiple NWB files. Some data from each file is allocated to calibration, minival, and evaluation splits as appropriate. 

### Code
This codebase contains starter code for implementing your own method for the FALCON challenge. 
- The `falcon_challenge` folder contains the logic for the evaluator. Submitted solutions must conform to the interface specified in `falcon_challenge.interface`.
- In `data_demos`, we provide notebooks that survey each dataset released as part of this challenge.
- In `decoder_demos`, we provide sample decoders and baselines that are formatted to be ready for submission to the challenge. To use them, see the comments in the header of each file ending in `_sample.py`. Your solutions should look similar once implemented! (Namely, you should have a `_decoder.py` file which conforms to `falcon_challenge.inferface` as well as a `_sample.py` file that indicates how your decoder class should be called.)

For example, you can prepare and evaluate a linear decoder by running:
```bash
python decoder_demos/sklearn_decoder.py --training_dir data/h1/held_in_calib/ --calibration_dir data/h1/held_out_calib/ --mode all --task h1
python decoder_demos/sklearn_sample.py --evaluation local --phase minival --split h1
```

Note: During evaluation, data file names are hashed into unique tags. Submitted solutions receive data to decode along with tags indicating the file from which the data originates in the call to their `reset` function. These tags are the keys of the the `DATASET_HELDINOUT_MAP` dictionary in `falcon_challenge/evaluator.py`. Submissions that intend to condition decoding on the data file from which the data comes should make use of these tags. For an example, see `fit_many_decoders` and `reset` in `decoder_demos/sklearn_decoder.py`.

### Docker Submission
To interface with our challenge, your code will need to be packaged in a Docker container that is submitted to EvalAI. Try this process by building and running the provided `sklearn_sample.Dockerfile`, to confirm your setup works. Do this with the following commands (once Docker is installed)
```bash
# Build
docker build -t sk_smoke -f ./decoder_demos/sklearn_sample.Dockerfile .
bash test_docker_local.sh --docker-name sk_smoke
```

For an example Dockerfile with annotations regarding the necessity and function of each line, see `decoder_demos/template.Dockerfile`.

## EvalAI Submission
Please ensure that your submission runs locally before running remote evaluation. You can run the previously listed commands with your own Dockerfile (in place of sk_smoke). This should produce a log of nontrivial metrics (evaluation is run on locally available minival).

To submit to the FALCON benchmark once your decoder Docker container is ready, follow the instructions on the [EvalAI submission tab]((https://eval.ai/web/challenges/challenge-page/2319/submission)). This will instruct you to first install EvalAI, then add your token, and finally push the submission. It should look something like:
`
evalai push mysubmission:latest --phase <phase-name> (dev or test)
`
(Note that you will not see these instruction unless you have first created a team to submit. The phase should contain a specific challenge identifier. You may need to refresh the page before instructions will appear.)

Please note that all submissions are subject to a 6 hour time limit. 

### Troubleshooting
Docker:
- If this is your first time with docker, note that `sudo` access is needed, or your user needs to be in the `docker` group. `docker info` should run without error.
- While `sudo` is sufficient for local development, the EvalAI submission step will ultimately require your user to be able to run `docker` commands without `sudo`.
- To do this, [add yourself to the `docker` group](https://docs.docker.com/engine/install/linux-postinstall/). Note you may [need vigr](https://askubuntu.com/questions/964040/usermod-says-account-doesnt-exist-but-adduser-says-it-does) to add your own user.

EvalAI:
- `pip install evalai` may fail on python 3.11, see: https://github.com/aio-libs/aiohttp/issues/6600. We recommend creating a separate env for submission in this case. 
