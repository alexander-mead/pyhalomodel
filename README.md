# Halo model
This repository is home to the code that comes with with halo-model review paper of Asgari, Heymans and Mead (2023). The software is written entirely in *python*, with extendability and reusability in mind. The purpose of this software is to take some of the drudgery out of performing basic calculations using the halo model. While the integrals that the halo model requires the researcher to evaluate are simple, in practice the changes of variables required to integrate halo profiles against halo mass functions can be confusing and tedious. In our experience this confusion has led to bugs and misunderstandings over the years, and our hope for this software is to reduce the proliferation of these somewhat. Our software can produce power spectra for *any* combinations of tracers, and simply requires halo profiles for the tracers to be specified. These could be matter profiles, galaxy profiles, or anything else, for example electron-pressure profiles (which pertain to the thermal Sunyaev-Zel`dovich effect). At the moment the halo profiles need to be specified in Fourier space, but we hope to allow for configuration-space halo profiles in the near future.

## Dependencies
These are the external libraries that you'll need to install for this pipe to run: 
* [CAMB](https://camb.readthedocs.io/en/latest/)
* [Dark Emulator](https://pypi.org/project/dark-emulator/)

## Installation
Simply clone the repository. You can then create an environment with all necessary dependacies using [poetry](https://python-poetry.org/):
```
poetry install --no-root
```
the `--no-root` option is required because the code is not currently installed as a global package, instead one simply uses the library via `import halomodel` with the `/halomodel` folder of the repository included in the `$PYTHONPATH` (see the notebooks for examples). Ensure you are working in the environment created by poetry when you do this.

You can also install without `poetry`, either system wide or using another environment manager. We include a `requirements.txt` for those that need it.

## Usage
Start a script with
```
import numpy
import halomodel
```
ensuring that `halomodel.py` is visible to your `path`. To make non-linear power spectrum predictions using the halo model requires a linear power spectrum. In our notebooks we always take this from `CAMB`, but it could come from any source. Calculations also require the variance in the linear density field when smoothed on some comoving scale $R$: $\sigma^2(R)$. Once again, this function could come from any source, but we take it from `CAMB`.

A typical call to create an instance of a `halo_model` object looks like
```
hmod = halomodel.halo_model(z, Omega_m, name='Tinker et al. (2010)', Dv=330., dc=1.686, verbose=True)
```
where:
- `z` is the redshift
- `Omega_m` is the cosmological matter density parameter (at $z=0$)
- `name` is the name of the halo mass function/linear halo bias pair to use
- `Dv` is the halo overdensity definition
- `dc` is the linear collapse threshold. 

Currently supported `name` choices are:
- `Press & Schecter (1974)`
- `Sheth & Tormen (1999)`
- `Sheth, Mo & Tormen (2001)`
- `Tinker et al. (2010)`
- `Despali et al. (2016)`

When the `hmod` instance is created the desired mass function is initialised. The code checks the choice of mass function against the `Dv` and `dc` values and will warn the user if there is an inconsistency.

To make a power-spectrum calculation one simply calls:
```
Pk_2h, Pk_1h, Pk_hm = hmod.power_spectrum(k, M, profiles, Pk_lin, sigmas)
```
where: 
- `k` is an array of comoving Fourier wavenumbers (units: $h\mathrm{Mpc}^{-1}$)
- `M` is an array of halo masses (units: $h^{-1}M_\odot$)
- `profiles` is a list of `halo_profile`s (see below; the list could contain a single entry)
- `Pk_lin` is a function that evaluates the linear power spectrum at a given `k`
- `sigmas` is an array of root-variance linear density values at Lagrangian scales corresponding to `M`

The function returns a tuple of `Pk_2h` (two-halo), `Pk_1h` (one-halo), and `Pk_hm` (full halo model; usually the sum) power at the chosen `k` values. The `power_spectrum` method computes all possible auto- and cross-spectra given the list of halo profiles. For example, if four profiles were in the list this would compute the four autospectra and six unique cross spectra. The returned `Pk` are therefore 3D `numpy` arrays, with the first two indices corresponding to the halo profile and the third index corresponding to `k`. For example, `Pk[0, 0, :]` would isolate the auto spectrum of the field corresponding to the first halo profile, at all `k` values.

Halo profiles are instances of the `halo_profile` class. These are initialised like:
```
profile = halomodel.halo_profile(k, M, N, Uk, norm, var, mass, discrete)
```
where
- `k` is an array of comoving Fourier wavenumbers (units: $h\mathrm{Mpc}^{-1}$)
- `M` is an array of halo masses (units: $h^{-1}M_\odot$)
- `N` is an array of (mean) profile amplitudes, coresponding to each `M`
- `Uk` is a 2D array of the *normalised* ($U(k\to0)=1$) Fourier halo profile at each `M` (first index) and `k` (second index) value
- `norm` is a float containing the field normalisation
- `var` is an array containing the variance in the profile amplitude at each `M`
- `mass` is a boolean telling the code if the profile corresponds to mass density
- `discrete` is a boolean telling the code if it dealing with a discrete tracer or not

The arrays `k` and `M` must correspond to those in the `hmod.power_spectrum` call. Some examples best illustrate how to create your own halo profiles:
```
matter_profile = halo_profile(k, M, M, Uk_matter, rho_matter, var=None, mass=True, discrete=False)
```
would create a matter profile. Here `Uk_matter` would be the normalised Fourier transform of a matter profile (e.g., an NFW profile), the amplitude of each profile is exactly `M` (because the haloes are the mass), but the field normalisation is `rho_matter` (which can be accessed via `hmod.rhom`) because the field we are interested in is matter overdensity. There is no variance in the amount of mass in a halo at fixed mass (obviously). We use `mass=True` and `discrete=False` to tell the code that the profile corresponds to mass and not to some discrete tracer. Note that in this case we would get exactly the same `halo_profile` if we fixed the profile amplitude as `N=M/rho_matter` and the field normalisation as `norm=1.`.
```
galaxy_profile = halo_profile(k, M, N_galaxy, Uk_galaxy, rho_galaxy, var=var_galaxy, mass=False, discrete=True)
```
would create a galaxy profile. Here `Uk_galaxy` would be the normalised Fourier transform of a galaxy profile (e.g., an isothermal profile). The amplitude of the profile is the mean galaxy-occupation number at each `M`: `N_galaxy`. The field is normalised by the mean galaxy density `rho_galaxy`. The variance in galaxy number at each `M` is `var_galaxy`. We tell the code that the tracers are not `mass` but that they are `discrete`. In the discrete-tracer case it is essential to split the profile amplitude from the field normalisation if the discreteness of the tracer is to be accounted for properly.

Note that *covariance* in the mean profile amplitude between two different tracers is not currently supported (this can be important in halo-occupation models where galaxies are split into centrals and satellites and the presence of a satellite galaxy is conditional on the halo first containing a central galaxy); we hope to include this in future. Also any spatial variance or covariance in halo profiles at fixed mass is not currently supported; we have no plans to include this in future.

## Notebooks
There are several jupyter notebooks in the `notebooks` folder giving examples of how to do your own halo-model predictions. The main one is `demo.ipynb`, which gives a run down of most of the features of our software package. As a bonus, we include the notebooks that produced (almost) all of the plots in the review paper.

## Citation
Please add a citation to `Asgari, Heymans and Mead (2023)` if you use this code.
