# NDT2 Sample Decoder Dockerfile

FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel
RUN /bin/bash -c "python3 -m pip install falcon_challenge --upgrade"

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
ADD ./local_data/ndt2_h1_sample.pth data/decoder.pth
ADD ./local_data/ndt2_zscore_h1.pt data/zscore.pt

# Add runfile
ADD ./decoder_demos/ndt2_sample.py decode.py
ADD ./decoder_demos/ndt2_decoder.py ndt2_decoder.py

ENV TRACK "h1"

# Don't touch
ENV EVAL_DATA_PATH "/evaluation_data"

# CMD specifies a default command to run when the container is launched.
# It can be overridden with any cmd e.g. sudo docker run -it my_image /bin/bash
CMD ["/bin/bash", "-c", \
    "python decode.py --evaluation $EVALUATION_LOC --model-path data/decoder.pth --zscore-path data/zscore.pt --phase $TRACK"]