# Evaluation server side
We simply mount the ground truth directory to the container and run the evaluation script.
TODO we should avoid publicizing this to prevent name conflicts
docker run -v data/:evaluation_data/ [other_options] -it sk_smoke
TODO how to figure what the submitted container name is?
