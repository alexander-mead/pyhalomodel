# Standard imports
import numpy as np
import sys
import unittest

# Third-party imports
import camb

# Projet imports
sys.path.append('./../')
import cosmology # TODO: Remove dependancy
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
sigma_8_set = True # If True uses the following value
sigma_8  = 0.8

# Redshifts
z = 0.

# CAMB
kmax_CAMB = 200.
zmax_CAMB = 10.

# Halo mass range [Msun/h]

# Halo model
Mmin = 1e7; Mmax = 1e17
nM = 129
dc = cosmology.dc_NakamuraSuto(Omega_c+Omega_b)
Dv = cosmology.Dv_BryanNorman(Omega_c+Omega_b)
halo_definition = 'Mvir'
halomodel_name = 'Sheth & Tormen (1999)'
concentration_name = 'Duffy et al. (2008)'

### ###

### CAMB ###

# Sets cosmological parameters in camb to calculate the linear power spectrum
pars = camb.CAMBparams()
wb, wc = Omega_b*h**2, Omega_c*h**2

# This function sets standard and helium set using BBN consistency
pars.set_cosmology(ombh2=wb, omch2=wc, H0=100.*h, mnu=m_nu, omk=Omega_k)
pars.set_dark_energy(w=w, wa=wa, dark_energy_model='ppf') 
pars.InitPower.set_params(As=As, ns=ns, r=0.)
pars.set_matter_power(redshifts=[z], kmax=kmax_CAMB) # Setup the linear matter power spectrum

# Scale 'As' to be correct for the desired 'sigma_8' value if necessary
if sigma_8_set:
    camb_results = camb.get_results(pars)
    sigma_8_init = (camb_results.get_sigma8()[[z].index(0.)]).item()
    scaling = (sigma_8/sigma_8_init)**2
    As *= scaling
    pars.InitPower.set_params(As=As, ns=ns, r=0.)

# Now get the linear power spectrum
Pk_lin = camb.get_matter_power_interpolator(pars, 
                                            nonlinear=False, 
                                            hubble_units=True, 
                                            k_hunit=True, 
                                            kmax=kmax_CAMB,
                                            var1=camb.model.Transfer_tot,
                                            var2=camb.model.Transfer_tot, 
                                            zmax=zmax_CAMB,
                                           )
Omega_m  = pars.omegam # Extract the matter density
Pk_lin = Pk_lin.P      # Single out the linear P(k) interpolator
camb_results = camb.get_results(pars)
sigma_8 = (camb_results.get_sigma8()[[z].index(0.)]).item()

### ###

class TestPower(unittest.TestCase):

    # Power-spectrum calculation
    @staticmethod
    def test_power():

        # Array of halo masses [Msun/h]
        Ms = np.logspace(np.log10(Mmin), np.log10(Mmax), nM)

        # Read benchmark
        infile = 'benchmarks/HMx_0_26_3_z0.0.dat'
        benchmark = np.loadtxt(infile)
        ks = benchmark[:, 0]
        Pk_benchmark = benchmark[:, 4]*2.*np.pi**2/ks**3

        # Initialise halo model
        hmod = halomodel.halo_model(z, Omega_m, name=halomodel_name, Dv=Dv, dc=dc)

        # Halo mass range [Msun/h] and Lagrangian radii [Mpc/h] corresponding to halo masses
        Rs = hmod.Lagrangian_radius(Ms)
        sigmaRs = cosmology.sigmaR(Rs, camb_results, integration_type='camb')[[z].index(z)]
        rvs = hmod.virial_radius(Ms)
        cs = halomodel.concentration(Ms, z, halo_definition=halo_definition)

        # Create a matter profile
        matter_profile = halomodel.matter_profile(ks, Ms, rvs, cs, hmod.Om_m)

        # Power spectrum calculation
        _, _, Pk = hmod.power_spectrum(ks, Ms, {'m': matter_profile}, lambda k: Pk_lin(z, k), sigmas=sigmaRs)

        # Save data
        outfile = 'results/HMx_0_26_3_z0.0.dat'
        data = np.column_stack((ks, Pk['m-m']))
        np.savetxt(outfile, data, header='k [h/Mpc]; P(k) [(Mpc/h)^3]')

        # Carry out test
        # TODO: Do assertion only after creating data
        np.testing.assert_array_almost_equal(Pk['m-m']/Pk_benchmark, 1., decimal=3)

### ###

### Unittest ###

if __name__ == '__main__':
    unittest.main()

### ###
