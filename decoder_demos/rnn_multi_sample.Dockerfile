# NDT2 Sample Decoder Dockerfile

FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-devel
RUN /bin/bash -c "python3 -m pip install falcon_challenge --upgrade"
# ADD ./falcon_challenge falcon_challenge
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

# H1
# ADD ./local_data/h1_single_oracle_rnn_*.ckpt \data/
# ADD ./local_data/ndt2_zscore_h1.pt data/zscore.pt
# ENV SPLIT "h1"
# ENV CONFIG_STEM falcon/h1/h1_100
# ENV ALL_PATHS "data/h1_single_oracle_rnn_39csa51t.ckpt  data/h1_single_oracle_rnn_7r3tsg42.ckpt  data/h1_single_oracle_rnn_ikg7qjfw.ckpt  data/h1_single_oracle_rnn_undb9ibb.ckpt  data/h1_single_oracle_rnn_z09y8rwu.ckpt data/h1_single_oracle_rnn_4o0rwcdk.ckpt  data/h1_single_oracle_rnn_8zh5jglv.ckpt  data/h1_single_oracle_rnn_l4mh9x4l.ckpt  data/h1_single_oracle_rnn_v1qq1woe.ckpt data/h1_single_oracle_rnn_5rdrk3db.ckpt  data/h1_single_oracle_rnn_dk37awax.ckpt  data/h1_single_oracle_rnn_n44ylznx.ckpt  data/h1_single_oracle_rnn_y1n2qdsi.ckpt"

# M1
# ADD ./local_data/m1_single_oracle_rnn_*.ckpt data/
# ADD ./local_data/ndt2_zscore_m1.pt data/zscore.pt
# ENV SPLIT "m1"
# ENV CONFIG_STEM falcon/m1/m1_100
# ENV ALL_PATHS "data/m1_single_oracle_rnn_45neu57a.ckpt  data/m1_single_oracle_rnn_9ea31gav.ckpt  data/m1_single_oracle_rnn_gw1er3az.ckpt  data/m1_single_oracle_rnn_y0s9f8nh.ckpt data/m1_single_oracle_rnn_73gm6s7v.ckpt  data/m1_single_oracle_rnn_aj76s8lj.ckpt  data/m1_single_oracle_rnn_s3880opv.ckpt"

# M2
ADD ./local_data/m2_single_oracle_rnn_*.ckpt data/
ADD ./local_data/ndt2_zscore_m2.pt data/zscore.pt
ENV SPLIT "m2"
ENV CONFIG_STEM falcon/m2/m2_100
ENV ALL_PATHS "data/m2_single_oracle_rnn_04yxz3ls.ckpt  data/m2_single_oracle_rnn_8en5p25l.ckpt  data/m2_single_oracle_rnn_k0e7g14s.ckpt  data/m2_single_oracle_rnn_q3kii5uw.ckpt data/m2_single_oracle_rnn_0ojd346m.ckpt  data/m2_single_oracle_rnn_e1xp8g1i.ckpt  data/m2_single_oracle_rnn_m3ezdb44.ckpt  data/m2_single_oracle_rnn_rzlc1bj9.ckpt"

# Add runfile
RUN pwd
ADD ./decoder_demos/ndt2_sample.py decode.py

ENV BATCH_SIZE 1
ENV PHASE "test"

# Make sure this matches the mounted data volume path. Generally leave as is.
ENV EVAL_DATA_PATH "/dataset/evaluation_data"
# ADD ./falcon_challenge falcon_challenge 

# CMD specifies a default command to run when the container is launched.
# It can be overridden with any cmd e.g. sudo docker run -it my_image /bin/bash
CMD ["/bin/bash", "-c", \
    "python decode.py --evaluation $EVALUATION_LOC --model-path '' --config-stem $CONFIG_STEM --zscore-path data/zscore.pt --split $SPLIT --batch-size $BATCH_SIZE --phase $PHASE --model-paths $ALL_PATHS"]