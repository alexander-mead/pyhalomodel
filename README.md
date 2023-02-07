# Halo model
This repository is home to the code that comes with with halo-model review paper of `Asgari, Mead & Heymans (2023)`. The software is written entirely in *python*, with extendability and reusability in mind. The purpose of this software is to take some of the drudgery out of performing basic calculations using the halo model. While the integrals that the halo model requires the researcher to evaluate are simple, in practice the changes of variables required to integrate halo profiles against halo mass functions can be confusing and tedious. In our experience this confusion has led to bugs and misunderstandings over the years, and our hope for this software is to reduce the proliferation of these somewhat. Our software can produce power spectra for *any* combinations of tracers, and simply requires halo profiles for the tracers to be specified. These could be matter profiles, galaxy profiles, or anything else, for example electron-pressure profiles (which pertain to the thermal Sunyaev-Zel`dovich effect).

## Dependencies
These are the external libraries that you'll need to install for this pipe to run: 
* [CAMB](https://camb.readthedocs.io/en/latest/)
* [Dark Emulator](https://pypi.org/project/dark-emulator/)

## Installation
Simply clone the repository. You can then create an environment with all necessary dependencies using [poetry](https://python-poetry.org/):
```
poetry install --no-root
```
the `--no-root` option is required because the code is not currently installed as a global package, instead one simply uses the library via `import halomodel` with the `/src` folder of the repository included in the `$PYTHONPATH` (see the notebooks for examples). Ensure you are working in the environment created by poetry when you do this.

You can also install without `poetry`, either system wide or using another environment manager. We include a `requirements.txt` for those that need it.

## Usage
Start a script with
```
import numpy
import halomodel
```
ensuring that `halomodel.py` is visible to your `path`. To make non-linear power spectrum predictions using the halo model requires a linear power spectrum. In our notebooks we always take this from `CAMB`, but it could come from any source. Calculations also require the variance in the linear density field when smoothed on comoving scale $R$: $\sigma^2(R)$. Once again, this function could come from any source, but we take it from `CAMB`.

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
- `profiles` is a dictionary of `halo_profile`s (see below; could contain a single entry)
- `Pk_lin` is a function that evaluates the linear power spectrum at a given `k`
- `sigmas` is an array of root-variance linear density values at Lagrangian scales corresponding to `M`

The function returns a tuple of `Pk_2h` (two-halo), `Pk_1h` (one-halo), and `Pk_hm` (full halo model; usually the sum) power at the chosen `k` values. The `power_spectrum` method computes all possible auto- and cross-spectra given the dictionary of halo profiles. For example, if three profiles were in the dictionary this would compute the three autospectra and three unique cross spectra. The returned `Pk` are then dictionaries containing all possible spectra. For example, if `halo_profiles={'a':profile_a, 'b':profile_b, 'c':profile_c}` then the `Pk` dictionaries will contain the keys: `a-a`; `a-b`; `a-c`; `b-a`; `b-b`; `b-c`; `c-a`; `c-b`; `c-c`, where the values in `a-b` and `b-a` (for example) will be identical. Of course, each value in the `Pk` dictionary is an array of the power at all `k` values.

Halo profiles are instances of the `halo_profile` class. These are initialised in Fourier space like:
```
halomodel.halo_profile(k, M, Uk, amp=None, norm=1., var=None, mass_tracer=False, discrete_tracer=False)
```
where
- `k` is an array of comoving Fourier wavenumbers (units: $h\mathrm{Mpc}^{-1}$)
- `M` is an array of halo masses (units: $h^{-1}M_\odot$)
- `Uk` is a 2D array of the Fourier halo profile at each `k` (first index) and `M` (second index) value
- `amp` is an array of (mean) profile amplitudes, corresponding to each `M`
- `norm` is a float containing the field normalisation
- `var` is an array containing the variance in the profile amplitude at each `M`
- `mass_tracer` is a boolean telling the code if the profile corresponds to mass density
- `discrete_tracer` is a boolean telling the code if it dealing with a discrete tracer or not

The arrays `k` and `M` must correspond to those in the `hmod.power_spectrum` call. If `amp=None` the Fourier profile is assumed to be normalised such that $U(k\to0, M)$ gives the total contribution of the halo to the field

Some examples best illustrate how to create your own halo profiles:
```
matter_profile = halomodel.halo_profile(k, M, Uk_matter, amp=M, norm=rho_matter, mass_tracer=True)
```
would create a matter profile. Here `Uk_matter` would be the normalised Fourier transform of a matter profile (e.g., an NFW profile), the amplitude of each profile is exactly `M` (because the haloes are the mass), but the field normalisation is `rho_matter` (which can be accessed via `hmod.rhom`) because the field we are interested in is matter overdensity. We use `mass_tracer=True` to tell the code that the profile corresponds to mass. Note that in this case we would get exactly the same `halo_profile` if we fixed the profile amplitude as `amp=M/rho_matter` and the field normalisation as `norm=1.`.
```
galaxy_profile = halomodel.halo_profile(k, M, Uk_galaxy, amp=N_galaxy, norm=rho_galaxy, var=var_galaxy, discrete_tracer=True)
```
would create a galaxy profile. Here `Uk_galaxy` would be the normalised Fourier transform of a galaxy profile (e.g., an isothermal profile). The amplitude of the profile is the mean galaxy-occupation number at each `M`: `amp=N_galaxy`. The field is normalised by the mean galaxy density `rho_galaxy`. The variance in galaxy number at each `M` is `var_galaxy`. We tell the code that `discrete_tracer=True` because in the discrete-tracer case it is essential to split the profile amplitude from the field normalisation if the discreteness of the tracer is to be accounted for properly.

Note that *covariance* in the mean profile amplitude between two different tracers is not currently supported (this can be important in halo-occupation models where galaxies are split into centrals and satellites and the presence of a satellite galaxy is conditional on the halo first containing a central galaxy); we hope to include this in future. Also any spatial variance or covariance in halo profiles at fixed mass is not currently supported; we have no plans to include this in future.

Halo profiles can also be specified in configuration (real) space, via a function of radius from the halo centre. This is slower than specifying the Fourier profiles since the conversion to Fourier space will need to be performed internally.
```
halomodel.halo_profile.configuration_space(k, M, Prho, rv, c, amp=None, norm=1., var=None, mass_tracer=False, discrete_tracer=False):
```
the arguments are similar to those for Fourier-space profiles, except that
- `Prho` is a the halo profile multiplied by $4\pi r^2$ with call signature `Prho(r, M, rv, c)`
- `rv` is the halo virial radius (units: $\mathrm{Mpc}/h$)
- `c` is the halo concentration

the multiplication by $4\pi r^2$ in `Prho` is to avoid singularities that often occur in halo profiles at $r=0$. Again, some examples best illustrate how to use this
```
def Prho_matter(r, M, rv, c):
   rs = rv/c
   return r/(1.+r/rs)**2.

matter_profile = halomodel.halo_profile.configuration_space(k, M, Prho_matter, rv, c, amp=M/rho_matter, mass_tracer=True)
```
note that because we specify the amplitude here we do not need to worry about constant factors in the `Prho` definition, since the profile normalisation will be calculated self consistently. Note also that because we set `amp=M/rho_matter` (matter *overdensity*) we can omit the `norm` argument, which defaults to `1.`.
```
Prho_gal = lambda r, M, rv, c: 1. # Isothermal profile

galaxy_profile = halomodel.halo_profile.configuration_space(k, M, Prho_gal, rv, c, amp=N_galaxy, norm=rho_galaxy, var=var_galaxy, discrete_tracer=True)
```
in the discrete tracer case it is important to split up `norm` and `amp` so that `amp` is something that can be interpreted as the mean of a discrete probability distribution.


## Notebooks
There are several jupyter notebooks in the `notebooks` folder giving examples of how to do your own halo-model predictions. The main one is `demo.ipynb`, which gives a run down of most of the features of our software package. As a bonus, we include the notebooks that produced (almost) all of the plots in the review paper.

## License


## Citation
Please add a citation to `Asgari, Mead & Heymans (2023)` if you use this code.
