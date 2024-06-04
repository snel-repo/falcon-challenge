# NoMAD Sample Decoder Dockerfile
# what base image should your container have for your code to run? 
FROM nvidia/cuda:11.2.2-base-ubuntu18.04

# Install base utilities
RUN apt-get update \
    && apt-get install -y wget

# Install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-py37_4.12.0-Linux-x86_64.sh
RUN bash Miniconda3-py37_4.12.0-Linux-x86_64.sh -b -p /miniconda
ENV PATH=$PATH:/miniconda/condabin:/miniconda/bin
RUN conda init
RUN conda install -c nvidia cudatoolkit=11.2
RUN conda install -c conda-forge cudnn=8.1

# this block stays the same
RUN /bin/bash -c "python -m pip install falcon_challenge --upgrade"
ADD ./falcon_challenge falcon_challenge
ENV PREDICTION_PATH "/submission/submission.csv"
ENV PREDICTION_PATH_LOCAL "/tmp/submission.pkl"
ENV GT_PATH = "/tmp/ground_truth.pkl"

# Users should install additional decoder-specific dependencies here.
RUN apt-get update && \
    apt-get install -y git
ADD ./lfads_tf2 lfads_tf2
RUN cd lfads_tf2 && \
    git checkout behavior && \
    pip install -e . && \
    pip install protobuf==3.20.0
ADD ./align_tf2 align_tf2

ENV EVALUATION_LOC "local"

# Add files from local context into Docker image
ADD ./nomad_baseline/submissions.yaml submissions.yaml
ADD ./nomad_baseline/submittable_models .

# Add runfile
RUN pwd
ADD ./decoder_demos/nomad_sample.py decode.py
ADD ./decoder_demos/nomad_decoder.py nomad_decoder.py

ENV SPLIT "m2"
ENV PHASE "minival"

# Make sure this matches the mounted data volume path. Generally leave as is.
ENV EVAL_DATA_PATH "/dataset/evaluation_data"

# CMD specifies a default command to run when the container is launched.
# It can be overridden with any cmd e.g. sudo docker run -it my_image /bin/bash
CMD ["/bin/bash", "-c", \
    "python decode.py --evaluation $EVALUATION_LOC --split $SPLIT --phase $PHASE --docker true"]