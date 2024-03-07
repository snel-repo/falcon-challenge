# Evaluation server side
We simply mount the ground truth directory to the container and run the evaluation script.
TODO we should avoid publicizing this to prevent name conflicts
docker run -v data/:evaluation_data/ [other_options] -it sk_smoke
e.g.
sudo docker run -v ~/projects/stability-benchmark/data:/evaluation_data/ -it sk_smoke /bin/bash
TODO Thiago -- script to wire and run the _submitted container_ and the _private evaluation dir_
