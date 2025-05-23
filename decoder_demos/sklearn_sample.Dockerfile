# Lightly commented Dockerfile example
# Dockerfile flow for EvalAI for a specific track

# The user prepares a Docker image with this file and decoder defined locally
# The user then directly pushes this image to EvalAI with the eval-ai cli.
# see e.g. https://eval.ai/web/challenges/challenge-page/1615/submission (You may have to register for the challenge)

# Base image specifies basic dependencies; if you're using TF/Jax, you may want to use a different base image.
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel
# RUN /bin/bash -c "python3 -m pip install falcon_challenge --upgrade"
RUN apt-get update && apt-get install -y git
RUN /bin/bash -c "python3 -m pip install falcon_challenge --upgrade"
# RUN pwd
ENV PREDICTION_PATH "/submission/submission.csv"
ENV PREDICTION_PATH_LOCAL "/tmp/submission.pkl"
ENV GT_PATH "/tmp/ground_truth.pkl"

# TODO ensure falcon_challenge available on dockerhub...

# Users should install additional decoder-specific dependencies here.

ENV EVALUATION_LOC remote
# ENV EVALUATION_LOC local

# Add files from local context into Docker image
# Note local context reference is the working dir by default, see https://docs.docker.com/engine/reference/commandline/build/

# Add ckpt
# Note that Docker cannot easily import across symlinks, make sure data is not symlinked
# ADD ./local_data/sklearn_FalconTask.h1.pkl data/decoder.pkl
# ADD ./local_data/sklearn_FalconTask.m1.pkl data/decoder.pkl
ADD ./local_data/sklearn_FalconTask.m2.pkl data/decoder.pkl

# Add source code/configs (e.g. can also be cloned)
ADD ./decoder_demos/ decoder_demos/
ADD ./data_demos/ data_demos/

# Add runfile
ADD ./decoder_demos/sklearn_sample.py decode.py
ADD ./decoder_demos/filtering.py filtering.py

# ENV SPLIT "h1"
# ENV SPLIT "m1"
ENV SPLIT "m2"
ENV PHASE "minival"
# ENV PHASE "test"
ENV BATCH_SIZE 4

# Make sure this matches the mounted data volume path. Generally leave as is.
ENV EVAL_DATA_PATH "/dataset/evaluation_data"

RUN pwd
# for local evaluation infra testing
# ADD ./falcon_challenge falcon_challenge

# CMD specifies a default command to run when the container is launched.
# It can be overridden with any cmd e.g. sudo docker run -it my_image /bin/bash
CMD ["/bin/bash", "-c", \
    "python decode.py --evaluation $EVALUATION_LOC --model-path data/decoder.pkl --split $SPLIT --phase $PHASE --batch-size $BATCH_SIZE"]