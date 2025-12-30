### Idea
In domains, where simmple thought  processes and sub solutions, can be easily verified and generated, one could use CLIP to encode and train on those domans to think and solve the sim2comsim problem.

### Experiments:
* RL Game - in the optimal case: 2D, 3D
* Arc-AGI: maybe? just for visual reasoning


* WOD - images + trajecotories + object tracking (=> simulation generation) => simulation prediction + trajectory generation
    Can we include spatial reasoning here?
    How does spatial reasoning work? - Could we build spatial reasoning with physics simulations?  - different datasets than WOD?

* WOD - baseline: imitation learning from trajectories + images => trajectory generation

### Pipeline:
1. Collect data from the domain - images + trajectories + object tracking
2. Generate simulations from the collected data through object tracking, physics engine, language reasoning, trajectory generation
3. Train a model to align collected data and generate simulations from collected data 

### Datasets



### Training


## To-Do's
* get code to run
* introduction
* methods










