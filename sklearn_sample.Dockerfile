# Lightly commented Dockerfile example
# Dockerfile flow for EvalAI for a specific track

# The user prepares a Docker image with this file and decoder defined locally
# The user then directly pushes this image to EvalAI with the eval-ai cli.
# see e.g. https://eval.ai/web/challenges/challenge-page/1615/submission (You may have to register for the challenge)

# Base image specifies basic dependencies; if you're using TF/Jax, you may want to use a different base image.
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel
# For GPU support, consider using NVIDIA's base image.
# FROM nvidia/cuda:11.0-cudnn8-devel-ubuntu20.04
RUN pwd
RUN /bin/bash -c "python3 -m pip install falcon_challenge"
# TODO ensure falcon_challenge available on dockerhub...

# Users should install additional decoder-specific dependencies here.

ENV EVALUATION_LOC remote

# Add files from local context into Docker image
# Note local context reference is the working dir by default, see https://docs.docker.com/engine/reference/commandline/build/

# Add ckpt
# Note that Docker cannot easily import across symlinks, make sure data is not symlinked
ADD ./local_data/sklearn_FalconTask.h1.pkl data/decoder.pkl

# Add source code/configs (e.g. can also be cloned)
ADD ./decoder_demos/ decoder_demos/

# Add runfile
ADD ./decode_sample.py decode.py

ENV TRACK "h1"

# Don't touch
ENV EVAL_DATA_PATH "/evaluation_data"

# CMD specifies a default command to run when the container is launched.
# It can be overridden with any cmd e.g. sudo docker run -it my_image /bin/bash
CMD ["/bin/bash", "-c", \
    "python decode.py --evaluation $EVALUATION_LOC --model-path data/decoder.pkl --phase $TRACK"]