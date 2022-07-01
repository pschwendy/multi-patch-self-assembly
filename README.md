# multi-patch-self-assembly

## Why is this here?
We are studying the ability the pore-forming host particles to capture guest particles, specifically, we are examining the effect of the size and shape of guest particles on their capture.
This is a model for a colloidal filtration/separation setup: we have particles that form open structures, but it's unclear what types of applications these systems can be used for.
That we have open structures suggests the possibility to selectively capture another species of particle (i.e., the guests), but it is currently unknown how the size and shape of the guest particle will affect the capture/loading behavior.
We hypothesize that, for a given size and shape of pores that form in the host structure, the size and shape of the guest particles will dictate their relative loading capacity.
For example, it seems obvious that particles that are too large to fit in the pores will not be captured at all, and that smaller particles will be captured more readily than larger ones.
We aim to quantify this behavior here.

## Simulation details
The simulations scripted in this repo start from a dilute mixture of patchy host and non-patchy guest particles.
The system is compressed to a higher density, but still fairly dilute, and then the temperature is slowly lowered (or equivalently, the strength of the patchy interactions is slowly increased).
We choose an annealing rate such that we get slow growth of the host structure to facilitate guest capture that more accurately reflects equilibrium; a rapid quench would cause rapid growth of multiple crystal grains with a guest capture that is heavily influenced by kinetics.

### Specific system compositions
We ultimately aim to quantify the selectivity of the guest capture in a competitive capture process: can we efficiently separate a mixture of different guest particles?
To answer this, we first need loading curves for systems containing a single species of guest particles.
If the pore loading is dependent upon the size and shape of the guest particles, then it is possible there will be at least some degree of selectivity in a competitive loading scenario.
If the pore loading is not dependent upon the size and shape of the guest particles, then it's a stretch to imagine there will be selectivity in the competitive loading process.
So the first set of simulations aims to quantify the guest capture in systems of square guest particles as function of their size.

If there is a clear trend of guest capture capacity with guest size, then we can predict the relative guest uptake in a mixture of guests of different sizes.
We will run guest capture simulations with a binary mixture of guest particles to test if the relative pore loading matches our expectations based on the noncompetitive scenario.

We ultimately aim to develop a model and/or heuristics to be able to predict the ability of the open-structure-forming hosts to separate a mixtures of guest particles of different size and shape.
The data generated from these simulations will guide our model development and allow us to test our predictions.

## What is included in this repo?
Simulation and anlysis code for systems of patchy polygons with non-patchy guest particles.
This repo is primarily 2 [signac-flow](https://signac.io/) projects: one for running the simulations and another for the analysis.
We only keep files that are necessary to generate the data in the repo; the simulation data (e.g., the simulation trajectories for each state point) are not kept under version control.
We do this for several reasons:

  1. The simulation data can become quite large (for version control, at least)
  1. The data space is still evolving, in that some jobs may be removed or reparameterized, and saving multiple snapshots of every state point will make the repo too big
  1. It is good practice to only keep what is needed to generate data in a repo
