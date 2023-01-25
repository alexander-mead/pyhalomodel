# Standard imports
import numpy as np
import sys

# Third-party imports
import camb

# Projet imports
sys.path.append('./../')
import cosmology
import halomodel

### Parameters ###

# Set cosmological parameters
Omega_c = 0.25
Omega_b = 0.05
Omega_k = 0.0
h = 0.7
As = 2e-9
ns = 0.96
w = -1.0
wa = 0.0
m_nu = 0.0 # [eV]

# You can choose to set sigma_8, in that case we scale the power spectrum
sigma_8_set = True # If True uses the following value
sigma_8  = 0.7

# Wavenumber range [h/Mpc]
kmin = 1e-3; kmax = 200
nk = 101
ks = np.logspace(np.log10(kmin), np.log10(kmax), nk)

# Redshifts
z = 0.
zmax = 2. # CAMB parameter

# Halo mass range [Msun/h]
Mmin = 1e9; Mmax = 1e15
nM = 129

### ###

### CAMB ###

# Sets cosmological parameters in camb to calculate the linear power spectrum
pars = camb.CAMBparams()
wb   = Omega_b*h**2
wc   = Omega_c*h**2

# This function sets standard and helium set using BBN consistency
pars.set_cosmology(ombh2=wb, omch2=wc, H0=100.*h, mnu=m_nu, omk=Omega_k)
pars.set_dark_energy(w=w, wa=wa, dark_energy_model='ppf') 
pars.InitPower.set_params(As=As, ns=ns, r=0)
pars.set_matter_power(redshifts=[z], kmax=kmax) # Setup the linear matter power spectrum

# Scale 'As' to be correct for the desired 'sigma_8' value if necessary
if sigma_8_set:
    camb_results = camb.get_results(pars)
    sigma_8_init = (camb_results.get_sigma8()[[z].index(0.)]).item()
    scaling = (sigma_8/sigma_8_init)**2
    As *= scaling
    pars.InitPower.set_params(As=As, ns=ns, r=0)

# Now get the linear power spectrum
Pk_lin = camb.get_matter_power_interpolator(pars, 
                                            nonlinear=False, 
                                            hubble_units=True, 
                                            k_hunit=True, 
                                            kmax=kmax,
                                            var1=camb.model.Transfer_tot,
                                            var2=camb.model.Transfer_tot, 
                                            zmax=zmax,
                                           )
Omega_m  = pars.omegam # Extract the matter density
Pk_lin = Pk_lin.P      # Single out the linear P(k) interpolator
camb_results = camb.get_results(pars)
sigma_8 = (camb_results.get_sigma8()[[z].index(0.)]).item()

### ###

### Halo model ###

# Halo mass range [Msun/h] over which to integrate

Ms = np.logspace(np.log10(Mmin), np.log10(Mmax), nM)

# Lagrangian radii [Mpc/h] corresponding to halo masses
Rs = cosmology.Lagrangian_radius(Ms, Omega_m)

### ###
