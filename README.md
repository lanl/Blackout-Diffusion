# Blackout-Diffusion

This repository host the source codes for the project "Blackout Diffusion: Generative Diffusion Models in Discrete-State Spaces", authored by Javier E. Santos, Zachary R. Fox (former CCS-3/LANL, currently at ORNL), Nicholas Lubbers (CCS-3, LANL), and Yen Ting Lin (CCS-3, LANL). The paper has been accepted by the 40th International Conference on Machine Learning. 

The source codes were reviewed and approved with a LANL C-number C23047 by the Richard P. Feynman Center for Innovation (FCI) at the Los Alamos National Laboratory.

## NCSN++ code base
The NCSN++ network, which was used to learn the reverse-time transition rates, was cloned from https://github.com/yang-song/score_sde_pytorch/tree/1618ddea340f3e4a2ed7852a0694a809775cf8d0

#### Differential changes to the SDE code base
- The file `util.py` and `datasets.py` was updated for our purpose 
- The file `loadDataPipeline.py` was added (for some reason, the original integrated data pipeline could not be used on our internal computational cluster.)

## Blackout Diffusion codes

Training and generation (inference) are separated into two Jupyter Notebooks. The first time when a training code is executed, the forward solution of the $\beta$-decay process at observation times is solved and stored in `forwardSolution.npz`. Later on, running the same training codes would only read the stored solution.

### CIFAR-10 dataset
- `blackout-cifar10-train-by-finiteTime.ipynb`:  Training by the finite-time formulation of the loss function
- `blackout-binarizedMNIST-train-by-instantaneous.ipynb`:  Training by the instantaneous formulation of the loss function
- `blackout-binarizedMNIST-generate-by-binomailBridge.ipynb`:  Generation using the binomial bridge formula
- `blackout-cifar10-generate-by-tauLeaping.ipynb`:  Generation using the $\tau$-leaping formulation

### Binarized MNIST dataset
- `blackout-binarizedMNIST-train-by-instantaneous.ipynb`:  Training by the instantaneous formulation of the loss function
- `blackout-binarizedMNIST-generate-by-binomailBridge.ipynb`:  Generation by the binomial bridge formula

### CelebA dataset
- `blackout-celebA64-train-by-instantaneous.ipynb`:  Training by the instantaneous formulation of the loss function
- `blackout-celebA64-generate-by-binomialBridge.ipynb`:  Generation by the binomial bridge formula



