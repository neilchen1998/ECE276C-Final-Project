# ECE276CFinalProject


## Abstract
In this project, our goal is to command a robotic arm that can hold a cup of water without spilling it out and move from a starting position to an end position.
This type of problem used to be done by generating samples randomly and projecting samples that are invalid and making them become valid. 
If we have many samples that are invalid, then many projections need to be done and that is very computational intensive. 
We want to speed up the process by utilizing a VAE model.

## Methods
We first implement our baseline PRM planner in OMPL, and pass it random points from the configuration space projected onto the constrained space using *scipy.optimize.minimize()*. 
We then use the samples generated in these runs to train a Variational Auto-Encoder to give us samples as close as possible to the constrained space.
We then run our improved PRM planner in OMPL using samples from our VAE.
We plan paths for two robots, a two link planar robot, and the Panda robot.

## Results
We tested the speed and effectiveness of both our Baseline and the VAE, as both a black box and as part of the OMPL framework. 
We also tested on two robots, first a 2DOF planar robot, then on the Panda 7DOF robot.

### Baseline
When planning for the two link robot with a constraint
threshold of π4 , the baseline reliably finds a path in around
1526 samples, with a runtime of 39.38 seconds. When gener-
ating the same number of samples on its own, without OMPL
embedding, the runtime is 2.489 seconds. The objective value
is 2.221.

### VAE
When planning for the two link robot with a constraint
threshold of π4 , the VAE reliably finds a path in 1508 samples,
with a runtime of 39.844 seconds. When generating the same
number of samples on its own, without OMPL embedding, the
runtime is 0.388. The objective value is 2.226.

## Conclusion
We found that using a precomputed machine learning model
to predict samples on a constrained space has the potential to
speed up PRM by reducing the amount of time spent projecting
samples. However, something about the interaction between
our python generators and the C++ OMPL implementation
caused this improvement to be lost.


Additionally, we found that the difference in speed extends
to both tighter constraints and higher dimensional configura-
tion spaces. This suggests that integrating a well designed and
well trained VAE model into the PRM sampling loop could
greatly speed up motion planning over constrained configura-
tion spaces, even when the mapping from the Cartesian space
constraint to the configuration space constraint is hard to find
analytically.
