# e-HAIL code sample
### Context
For my Ph.D. I study stochastic circuits and their ability to implement algorithms like neural networks directly in hardware. The circuits are stochastic, and so we are interested in their average behavior and accuracy. I developed a Monte Carlo simulation codebase in Python to study these features and part of this code is shared with you.

The code will probably not execute because it may try to access files stored only on my system, but this `README` will explain the code and give insight into my coding experience.

### Main simulation file:
`exp19_nonpow2_weighted_adders.py` is a main simulation file supported by the other included files. 

### Why this code
I chose this code for a few reasons:

1. It demonstrates my experience with workflow management with [signac](https://signac.io), a Python library. I use this software to track how various stochastic circuit configurations perform with various network hyperparameters. Signac seems similar in some respects to the MLOps software [weights and biases](https://wandb.ai/site) mentioned in the job posting. Anytime you see a `job` variable in the code, it is accessing simulation data (`job.data`) or information (`job.doc`) stored using signac’s automated workflow management system.
2. It demonstrates my proficiency with [PyTorch](https://pytorch.org), a leading Python deep learning library. The file `models03_updated_FMNIST.py` gives example of implementations of neural networks from research papers. `PSA_MLP2` is our custom network based on prior work. Then `helpers/binarized_modules.py` gives an example of custom PyTorch neural network layers (`SamplingLayer` and `PSA_Linear`) that I implemented to work specifically with our stochastic hardware approach.
3. It gives examples of object-oriented design. The `SCython/SNG_torch` folder contains implementations of certain circuit components. For example, our circuits can use various random number sources (RNSes). The `SCython/SNG_torch/RNS.py` file implements an abstract `RNS` class that serves as a template for any specific RNS we implement. The remainder of `RNS.py` contains examples of specific RNS implementations that inherit from the base `RNS` class. Ultimately, the object-oriented design combined with signac workflow management enables a very efficient method of experimentation that allowed us to run many simulations. It led to a robust understanding of our circuits and allows us to test new ideas with ease.
4. It gives some examples of documentation. This code is not shared with anyone and so the documentation may be a bit sloppy than what I would write when coding collaboratively.
5. There’s also some examples of visualization code in `exp19_nonpow2_weighted_adders.py`. It is very sloppy because there is little value in cleaning up this specific part of the code in an academic setting. My resume has two links to examples of plots I’ve made.

### Why I’m proud of code:

It is satisfying to see years of experience result in an efficient system. The SCython folder is the core of my simulation codebase that I import and use in almost every simulation file like `exp19_nonpow2_weighted_adders.py`. When I have a new research idea, it is amazing how quickly it can be simulated using this code, both in terms of the time it takes to write a new simulation file and in terms of how quickly the simulation runs. This ease of experimentation is especially appreciated as we work against tight conference deadlines where some of the best ideas appear late into the paper writing process.
