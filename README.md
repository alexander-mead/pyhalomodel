# Halo model
This repository is home to the code that comes with with halo-model review paper of Asgari, Heymans and Mead (2023). The software is written entirely in *python*, with extendability and reusability in mind. The purpose of this software is to take some of the drudgery out of performing basic calculations using the halo model. While the integrals that the halo model requires the researcher to evaluate are simple, in practice the changes of variables required to integrate halo profiles against halo mass functions can be confusing and tedious. In our experience this confusion has led to bugs and misunderstandings over the years, and our hope for this software is to reduce the proliferation of these somewhat. Our software can produce power spectra for *any* combinations of tracers, and simply requires halo profiles for the tracers to be specified. These could be matter profiles, galaxy profiles, or anything else, for example electron-pressure profiles (which pertain to the thermal Sunyaev-Zel`dovich effect). At the moment the halo profiles need to be specified in Fourier space, but we hope to allow for configuration-space halo profiles in the near future.

## Dependencies
These are the external libraries that you'll need to install for this pipe to run: 
* [CAMB](https://camb.readthedocs.io/en/latest/)
* [Dark Emulator](https://pypi.org/project/dark-emulator/)

## Installation
Simply clone the repository...

## Usage
To make non-linear power spectrum predictions using the halo model requires a linear power spectrum. In our notebooks we always take this from `CAMB`, but it could come from any source. Calculations also require the variance in the linear density field when smoothed on some comoving scale $R$: $\sigma^2(R)$. Once again, this function could come from any source, but we take it from `CAMB`. 

## Notebooks
There are several jupyter notebooks in the `notebooks` folder giving examples of how to do your own halo-model predictions. The main one is `demo.ipynb`, which gives a run down of most of the features of our software package. As a bonus, we include the notebooks that produced (almost) all of the plots in the review paper.

## Citation
Please add a citation to `Asgari, Heymans and Mead (2023)` if you use this code.

