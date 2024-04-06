# NDT2 Sample Decoder Dockerfile

FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel
RUN /bin/bash -c "python3 -m pip install falcon_challenge --upgrade"
ADD ./falcon_challenge falcon_challenge
ENV PREDICTION_PATH "/submission/submission.csv"
ENV PREDICTION_PATH_LOCAL "/tmp/submission.pkl"
ENV GT_PATH = "/tmp/ground_truth.pkl"

# Users should install additional decoder-specific dependencies here.
RUN apt-get update && \
    apt-get install -y git
RUN pwd
RUN git clone https://github.com/joel99/context_general_bci.git
RUN cd context_general_bci && \
    python3 -m pip install -e . --extra-index-url https://download.pytorch.org/whl/cu117
# RUN /bin/bash -c "git clone https://github.com/joel99/context_general_bci.git && cd context_general_bci && python3 -m pip install . && cd .."
# Install the Python package

ENV EVALUATION_LOC remote

# Add files from local context into Docker image
# Note local context reference is the working dir by default, see https://docs.docker.com/engine/reference/commandline/build/

# Add ckpt
# Note that Docker cannot easily import across symlinks, make sure data is not symlinked
ADD ./local_data/ndt2_h1_sample_nokey.pth data/decoder.pth
ADD ./local_data/ndt2_zscore_h1.pt data/zscore.pt

# Add runfile
RUN pwd
RUN pwd
ADD ./decoder_demos/ndt2_sample.py decode.py
ADD ./decoder_demos/ndt2_decoder.py ndt2_decoder.py

ENV SPLIT "h1"
ENV PHASE "test"

# Make sure this matches the mounted data volume path. Generally leave as is.
ENV EVAL_DATA_PATH "/dataset/evaluation_data"
# ADD ./falcon_challenge falcon_challenge 

# CMD specifies a default command to run when the container is launched.
# It can be overridden with any cmd e.g. sudo docker run -it my_image /bin/bash
CMD ["/bin/bash", "-c", \
    "python decode.py --evaluation $EVALUATION_LOC --model-path data/decoder.pth --zscore-path data/zscore.pt --split $SPLIT --phase $PHASE"]