# Technical elements missing
- How do we wire ground truth without literally placing in base image?
- Where does the evaluation server live, and how does it interact with the submitted Docker image?
- How do we get EvalAI to run the DockerImage (think some email needs to be made)
- Docker Image base.
- Metric implementations (maybe, or BK/JY insert)
- Make track dependent logic more rigorous
- Write documentation for user

## Group decisions
- Trial-batched inference? Force causal prediction?
    - What are the dimensions of the input to the prediction step?