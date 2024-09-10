# This is a stripped-down version of the sklearn_sample.Dockerfile in the same directory. 
# This file is intended to be a template for users to create their own Dockerfiles for EvalAI submissions.
# Comments are provided to indicate which lines are mandatory for a valid submission and which are submission-specific.

# NOTE: The environment variables PREDICTION_PATH, PREDICTION_PATH_LOCAL, GT_PATH, , and EVAL_DATA_PATH are used by the evaluator and should not be changed.
# The other environment variables could be hardcoded in the runfile, but it is recommended to set them in the Dockerfile for clarity.

# Dockerfile flow for EvalAI for a specific track

# The user prepares a Docker image with this file and decoder defined locally
# The user then directly pushes this image to EvalAI with the eval-ai cli.
# see e.g. https://eval.ai/web/challenges/challenge-page/1615/submission (You may have to register for the challenge)

# RECOMMENDED: Use a base Docker image that has the necessary dependencies for your decoder. The one provided is a good choice for PyTorch-based decoders.
# Note: If you're using TF/Jax, you may want to use a different base image.
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel

# MANDATORY: Leave these lines as they are. This will install an up-to-date copy of the falcon-challenge package into the Docker image.
RUN apt-get update && apt-get install -y git

RUN git clone https://github.com/snel-repo/falcon-challenge.git
RUN pip install -e falcon-challenge

# MANDATORY: Leave these environment variables as they are. This is how EvalAI will find the predictions and answers during evaluation.
ENV PREDICTION_PATH "/submission/submission.csv"
ENV PREDICTION_PATH_LOCAL "/tmp/submission.pkl"
ENV GT_PATH "/tmp/ground_truth.pkl"

# OPTIONAL: Users should install additional decoder-specific dependencies here (e.g. RUN pip install package_name)
RUN git clone https://github.com/pabloslash/EnSongdec.git
RUN pip install -e EnSongdec

# Forked from facebookresearch and adapted
RUN git clone https://github.com/pabloslash/encodec.git
RUN pip install -e encodec

RUN git clone https://github.com/pabloslash/songbirdcore.git
RUN pip install -e songbirdcore


# MANDATORY: Set EVALUATION_LOC. Use "remote" for submissions to EvalAI and "local" for local testing. This is passed to the evaluator in the runfile.
ENV EVALUATION_LOC remote

# Add files from local context into Docker image
# Note local context reference is the working dir by default, see https://docs.docker.com/engine/reference/commandline/build/

# MANDATORY: Add the decoder file to the Docker image. This should be the path to the decoder file in your local directory.
# Note that Docker cannot easily import across symlinks; make sure data is not symlinked
ADD b1_models/z_r12r13_21_2021.06.27_held_in_calib.nwb_FFNN_20240604_092705.pt data/decoder.pt

# OPTIONAL: Add any additional files that your decoder needs to the Docker image. The lines below are relevant to the ensongdec decoder.
ADD b1_models/z_r12r13_21_2021.06.27_held_in_calib.nwb_FFNN_20240604_092705_metadata.json data/decoder_config.json

# MANDATORY:  Add the runfile to the Docker image. This should be the path to the runfile in your local directory. This line is relevant to the sklearn_sample decoder.
ADD ensongdec_sample.py decode.py

# MANDATORY: Set SPLIT. This specifies the dataset that your decoder is being evaluated on. This is passed to the evaluator in the runfile.
ENV SPLIT "b1"

# MANDATORY: Set PHASE. "minival" should be used for local testing and submission to the minival phase on EvalAI. "test" should be used for submissions to the test phase.
# The minival set is a small subset of the held-in sessions, and is used for local testing and debugging. The test set is the full evaluation set for all sessions.
# Note that the test set is only available remotely on EvalAI. The minival set is available locally and remotely on EvalAI.
# This is passed to the evaluator in the runfile.
ENV PHASE "minival"

# OPTIONAL: Set BATCH_SIZE for batched evaluation. Batching accelerates evaluation by processing multiple samples at once.
# Note: the batch size is an attribute of the decoder. For the example sklearn_sample decoder, the batch size is passed to the decoder's constructor in the runfile.
ENV BATCH_SIZE 1

# MANDATORY: Leave this line as it is. This is a path to the evaluation data in the remote EvalAI environment.
ENV EVAL_DATA_PATH "/dataset/evaluation_data"

# MANDATORY: Specify a command for the container to run when it is launched. This command should run the decoder on the evaluation data.
# It can be overridden with any cmd e.g. sudo docker run -it my_image /bin/bash
CMD ["/bin/bash", "-c", \
    "python decode.py --evaluation $EVALUATION_LOC --model-paths data/decoder.pt --model-cfg-paths data/decoder_config.json --split $SPLIT --phase $PHASE --batch-size $BATCH_SIZE"]
