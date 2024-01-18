# Lightly commented Dockerfile example
# Dockerfile flow for EvalAI for a specific track

# The user prepares a Docker image with this file and decoder defined locally
# The user then directly pushes this image to EvalAI with the eval-ai cli.
# see e.g. https://eval.ai/web/challenges/challenge-page/1615/submission (You may have to register for the challenge)

# TODO - Challenge team needs to upload a base image to dockerhub. I think this just has base eval dependencies e.g. numpy, but double-check
FROM fairembodied/habitat-challenge:testing_2020_habitat_base_docker

# Users should install additional decoder-specific dependencies here.
RUN /bin/bash -c ". activate habitat; conda install pytorch torchvision torchaudio cudatoolkit=10.1 -c pytorch; conda install tensorboard; pip install ifcfg"

# Copy specific eval libs
RUN /bin/bash -c "git clone anything_else_you_need_thats_not_in_the_base_image"

RUN pwd

ENV AGENT_EVALUATION_TYPE remote

# These files come from local context e.g. by default, just the working directory, see https://docs.docker.com/engine/reference/commandline/build/
# TODO do we need to make these Docker dirs?
ADD ckpts/my_ckpt.pth ckpts/MODEL_CKPT.pth

# Challenge
ADD decode.py decode.py
# User defines their model source code or pulls it in however, here we add a locally defined src directory (user should provide)
ADD src/ src/

# User can configure their own config file here
ENV AGENT_CONFIG_DOCKER_PATH "configs/my_decoder.yaml"
ADD configs/my_decoder.yaml $AGENT_CONFIG_DOCKER_PATH

ENV TRACK "stability_23_human_7dof"
ENV TRACK_CONFIG_FILE "bci_stability_challenge/${TRACK}.yaml"
ADD configs/${TRACK}.yaml $TRACK_CONFIG_FILE

# Note - user is allowed to modify these however they want
# Docker runs this command once image is launched -- we need to ensure wiring is in place so that the user's code is run properly
# ? Where do we pipe in the "private" ground truth?
CMD ["/bin/bash", "-c", "source activate habitat && export PYTHONPATH=/evalai-remote-evaluation:$PYTHONPATH && export CHALLENGE_CONFIG_FILE=$TRACK_CONFIG_FILE && python decode.py --evaluation $AGENT_EVALUATION_TYPE  --model-path ckpts/MODEL_CKPT.pth --config-path $AGENT_CONFIG_DOCKER_PATH"]
# TODO - We should provide guidance that the compiled docker image for this config goes to --phase test_$TRACK