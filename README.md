# `pyhalomodel`

![image](https://user-images.githubusercontent.com/9140961/228342339-e1f908c7-e2ac-46c2-bb2a-1878bbfa3cf0.png)

This repository is home to the `pyhalomodel` package, which was written as part of the [Asgari, Mead & Heymans (2023)](https://arxiv.org/abs/2303.08752) halo-model review paper. The software is written entirely in `Python`, with extendability and reusability in mind. The purpose of this software is to take some of the drudgery out of performing basic calculations using the halo model. While the integrals that the halo model requires the researcher to evaluate are simple, in practice the changes of variables required to integrate halo profiles against halo mass functions can be confusing and tedious. In our experience this confusion has led to bugs and misunderstandings over the years, and our hope for this software is to reduce the proliferation of these somewhat. Our software can produce power spectra for *any* combinations of tracers, and simply requires halo profiles for the tracers to be specified. These could be matter profiles, galaxy profiles, or anything else, for example electron-pressure or HI profiles.

You might also be interested in this pure `Python` implementation of [HMcode](https://github.com/alexander-mead/HMcode-python), which makes use of the `pyhalomodel` package.

## Dependencies
* `numpy`
* `scipy`

## Installation
For the general user, `pyhalomodel` can be installed using `pip`:
```bash
> pip install pyhalomodel
```
If you you want to modify the source, or use the demo notebooks, then simply clone the repository. You can then create an environment with all necessary dependencies using [poetry](https://python-poetry.org/). From the cloned `pyhalomodel` directory:
```bash
> poetry install
```
The demo notebooks require some additional dependencies: `camb`; `dark-emulator`; `ipykernel`; `matplotlib`; `halomod`. These will be installed in the environment automatically. You can also install without `poetry`, either system wide or using another environment manager; we include a `requirements.txt`.

## Usage
Start a script with
```python
import numpy as np
import pyhalomodel as halo
```
Importing via `import pyhalomodel as halo` is nice because the functions and classes then have readable names (e.g., `halo.model`, `halo.profile`). To make non-linear power spectrum predictions using the halo model requires a linear power spectrum. In our demonstration notebooks we always take this from `CAMB`, but it could come from any source. Calculations also require the variance in the linear density field when smoothed on comoving scale $R$: $\sigma^2(R)$. Once again, this function could come from any source, but we take it from `CAMB`.

A typical call to create an instance of a `model` object looks like
```python
model = halo.model(z, Omega_m, name='Tinker et al. (2010)', Dv=330., dc=1.686, verbose=True)
```
where:
- `z` is the redshift
- `Omega_m` is the cosmological matter density parameter (at $z=0$)
- `name` is the name of the halo mass function/linear halo bias pair to use
- `Dv` is the halo overdensity definition
- `dc` is the linear collapse threshold

Currently supported `name` choices are:
- `Press & Schecter (1974)`
- `Sheth & Tormen (1999)`
- `Sheth, Mo & Tormen (2001)`
- `Tinker et al. (2010)`
- `Despali et al. (2016)`

When the `model` instance is created the desired mass function is initialised.

To make a power-spectrum calculation one simply calls:
```python
Pk_2h, Pk_1h, Pk_hm = model.power_spectrum(k, Pk_lin, M, sigmaM, profiles)
```
where: 
- `k` is an array of comoving Fourier wavenumbers (units: $h\mathrm{Mpc}^{-1}$)
- `Pk_lin` is an array of linear power spectrum values at a given `k` (units: $(h^{-1}\mathrm{Mpc})^3$)
- `M` is an array of halo masses (units: $h^{-1}M_\odot$)
- `sigmaM` is an array of root-variance linear density values at Lagrangian scales corresponding to `M`
- `profiles` is a dictionary of halo `profile`s (which could contain a single entry)

The function returns a tuple of `Pk_2h` (two-halo), `Pk_1h` (one-halo), and `Pk_hm` (halo model) power at the chosen `k` values. The `power_spectrum` method computes all possible auto- and cross-spectra given the dictionary of halo profiles. For example, if three profiles were in the dictionary this would compute the three autospectra and the three unique cross spectra. The returned `Pk` are then dictionaries containing all possible spectra. For example, if `profiles={'a':profile_a, 'b':profile_b, 'c':profile_c}` then the `Pk` dictionaries will contain the keys: `a-a`; `a-b`; `a-c`; `b-b`; `b-c`, `c-c`. It will also contain symmetric combinations (e.g., `b-a` as well as `a-b`) but the values will be identical. Each value in the `Pk` dictionary is an array of the power at all `k`.

Halo profiles are instances of the `profile` class. These are initialised in Fourier space like:
```python
profile = halo.profile.Fourier(k, M, Uk, amplitude=None, normalisation=1., variance=None, mass_tracer=False, discrete_tracer=False)
```
where
- `k` is an array of comoving Fourier wavenumbers (units: $h\mathrm{Mpc}^{-1}$)
- `M` is an array of halo masses (units: $h^{-1}M_\odot$)
- `Uk` is a 2D array of the Fourier halo profile at each `k` (first index) and `M` (second index) value
- `amplitude` is an array of (mean) profile amplitudes, corresponding to each `M`
- `normalisation` is a float containing the field normalisation
- `variance` is an array containing the variance in the profile amplitude at each `M`
- `mass_tracer` is a boolean telling the code if the profile corresponds to mass density
- `discrete_tracer` is a boolean telling the code if it dealing with a discrete tracer or not

The arrays `k` and `M` be identical to those in the subsequent `model.power_spectrum` call. If `amplitude=None` the Fourier profile is assumed to be normalised such that $U(k\to0, M)$ gives the total contribution of the halo to the field. Otherwise the profile is renormalised by the `amplitude`, and $U(k\to0, M)=1$ is assumed.

Some examples best illustrate how to create your own halo profiles:
```python
matter_profile = halo.profile.Fourier(k, M, Uk_matter, amplitude=M, normalisation=rho_matter, mass_tracer=True)
```
would create a matter profile. Here `Uk_matter` would be the normalised Fourier transform of a matter profile (e.g., an NFW profile), the amplitude of each profile is exactly `M` (because the haloes are the mass), but the field normalisation is `rho_matter` (which can be accessed via `model.rhom`) because the field we are usually interested in is matter *overdensity*. We use `mass_tracer=True` to tell the code that the profile corresponds to mass. Note that in this case we would get identical behaviour if we fixed the profile amplitude as `amplitude=M/rho_matter` and the field normalisation as `normalisation=1.`.
```python
galaxy_profile = halo.profile.Fourier(k, M, Uk_galaxy, amplitude=N_galaxy, normalisation=rho_galaxy, variance=var_galaxy, discrete_tracer=True)
```
would create a galaxy profile. Here `Uk_galaxy` would be the normalised Fourier transform of a galaxy profile (e.g., an isothermal profile). The amplitude of the profile is the mean galaxy-occupation number at each `M`: `N_galaxy`. The field is normalised by the mean galaxy density: `rho_galaxy`. For a given assumption about the mean galaxy-occupation number and halo model this can be calculated using the `average` method of the `model` class:
```python
rho_galaxy = hmod.average(M, sigmaM, N_galaxy)
```
The variance in galaxy number at each `M` is `var_galaxy`. If one is assuming Poisson statistics then `variance=N_galaxy` is appropriate, but any value can be used in principle, including `variance=None`, which ignores the contribution of tracer-occupation variance to the power. We tell the code that `discrete_tracer=True` because in the discrete-tracer case it is essential to split the profile amplitude from the field normalisation if the discreteness of the tracer is to be accounted for properly.

Halo profiles can also be specified in configuration (real) space, via a function of radius from the halo centre. This is slower than specifying the Fourier profiles since the conversion of the profile to Fourier space will need to be performed internally.
```python
halo.profile.configuration(k, M, rv, c, differential_profile, amplitude=None, normalisation=1., variance=None, mass_tracer=False, discrete_tracer=False):
```
the arguments are similar to those for Fourier-space profiles, except that
- `differential_profile` is a the halo profile multiplied by $4\pi r^2$ with call signature `differential_profile(r, M, rv, c)`
- `rv` is the halo virial radius (units: $h^{-1}\mathrm{Mpc}$)
- `c` is the halo concentration

the differential halo profile is the function defined such that integrating in radius between $0$ and $r_\mathrm{v}$ gives the total contribution of an individual halo to the field. It is usually the standard density profile multiplied $4\pi r^2$. This convention is used so as to avoid singularities that often occur in halo profiles at $r=0$. Again, some examples best illustrate how to use this
```python
def differential_profile_matter(r, M, rv, c):
    # This is NFW (1./((r/rs)*(1.+r/rs)**2)) multiplied by 4pir^2 with constant factors ignored
    rs = rv/c
    return r/(1.+r/rs)**2.

matter_profile = halo.profile.configuration(k, M, rv, c, differential_profile_matter, amplitude=M/rho_matter, mass_tracer=True)
```
note that because we specify the amplitude here we do not need to worry about constant factors in the `differential_profile` definition, since the profile normalisation will be calculated self consistently. Note also that because we set `amplitude=M/rho_matter` (matter *overdensity*) we can omit the `normalisation` argument, which defaults to `1.`.
```python
# Isothermal profile: 1/r^2, multiplied by 4pir^2 with constant factors ignored
differential_profile_gal = lambda r, M, rv, c: 1. 

galaxy_profile = halo.profile.configuration(k, M, rv, c, differential_profile_gal, amplitude=N_galaxy, normalisation=rho_galaxy, discrete_tracer=True)
```
in the discrete tracer case it is important to split up `normalisation` and `amplitude` so that `amplitude` is something that can be interpreted as the mean of a discrete probability distribution. In this example we have also decided to ignore the contribution of the variance in the number of galaxies at fixed halo mass to the eventual power spectrum calculation.

Note that the *covariance* in the mean profile amplitude between two different tracers is not currently supported. This can be important in halo-occupation models where galaxies are split into centrals and satellites and the presence of a satellite galaxy is conditional on the halo first containing a central galaxy. We hope to include this in future. Also any *spatial* variance or covariance in halo profiles at fixed mass is not currently supported; we have no plans to include this in future.

## Notebooks
There are several `jupyter` notebooks in the `notebooks/` directory giving examples of how to use the code. The first one to try is `demo-basic.ipynb`, which gives an overview of the main features of `pyhalomodel`. As a bonus, we include notebooks that produce (almost) all of the plots from the review paper.

## Citation
Please add a citation to [Asgari, Mead & Heymans (2023)](https://arxiv.org/abs/2303.08752) if you use this code.
