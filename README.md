# Blackout-Diffusion

This repository host the source codes for the project "Blackout Diffusion: Generative Diffusion Models in Discrete-State Spaces", authored by Javier E. Santos, Zachary R. Fox (former CCS-3/LANL, currently at ORNL), Nicholas Lubbers (CCS-3, LANL), and Yen Ting Lin (CCS-3, LANL). The paper has been accepted by the 40th International Conference on Machine Learning. 

The source codes were reviewed and approved with a LANL C-number C23047 by the Richard P. Feynman Center for Innovation (FCI) at the Los Alamos National Laboratory.

## NCSN++ code base
The NCSN++ network, which was used to learn the reverse-time transition rates, was cloned from https://github.com/yang-song/score_sde_pytorch/tree/1618ddea340f3e4a2ed7852a0694a809775cf8d0

## Differential changes to the SDE code base
- The file `util.py` and `datasets.py` was updated for our purpose 
- The file `loadDataPipeline.py` was added (for some reason, the original integrated data pipeline could not be used on our internal computational cluster.)
