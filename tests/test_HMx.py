# Standard imports
import numpy as np
import sys
import unittest

# Third-party imports
import camb

# Projet imports
sys.path.append('./../src')
import cosmology
import halomodel as halo

### Parameters ###

# Set cosmological parameters
Omega_c = 0.25
Omega_b = 0.05
Omega_k = 0.
h = 0.7
As = 2e-9
ns = 0.96
w = -1.
wa = 0.
m_nu = 0. # [eV]
sigma_8_set = True # If True uses the following value
sigma_8  = 0.8

# Redshifts
zs = [3., 2., 1., 0.5, 0.]

# CAMB
kmax_CAMB = 200.
zmax_CAMB = 10.

# Halo mass range [Msun/h]

# Halo model
Mmin, Mmax = 1e7, 1e17 # HMx defaults
nM = 129 # HMx defaults
Ms = np.logspace(np.log10(Mmin), np.log10(Mmax), nM)
halo_definition = 'Mvir'
concentration_name = 'Duffy et al. (2008)'
halomodel_names = {
    'Press & Schecter (1974)': 27,
    'Sheth & Tormen (1999)': 3,
    'Sheth, Mo & Tormen (2001)': 146,
    'Tinker et al. (2010)': 23,
    'Despali et al. (2016)': 87,
}

### ###

### CAMB ###

# Sets cosmological parameters in camb to calculate the linear power spectrum
pars = camb.CAMBparams()
wb, wc = Omega_b*h**2, Omega_c*h**2

# This function sets standard and helium set using BBN consistency
pars.set_cosmology(ombh2=wb, omch2=wc, H0=100.*h, mnu=m_nu, omk=Omega_k)
pars.set_dark_energy(w=w, wa=wa, dark_energy_model='ppf') 
pars.InitPower.set_params(As=As, ns=ns, r=0.)
pars.set_matter_power(redshifts=zs, kmax=kmax_CAMB) # Setup the linear matter power spectrum

# Scale 'As' to be correct for the desired 'sigma_8' value if necessary
if sigma_8_set:
    camb_results = camb.get_results(pars)
    sigma_8_init = (camb_results.get_sigma8()[zs.index(0.)]).item()
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
Pk_lin = Pk_lin.P # Single out the linear P(k) interpolator
camb_results = camb.get_results(pars)
sigma_8 = (camb_results.get_sigma8()[zs.index(0.)]).item()

# Loop over halo models
for halomodel_name, halomodel_number in halomodel_names.items():

    # Read benchmark
    file = 'HMx_26_'+str(halomodel_number)+'.dat'
    infile = 'benchmarks/'+file
    benchmark = np.loadtxt(infile)
    ks = benchmark[:, 0] # Wavenumbers
    Pk_benchmark = np.dot(benchmark[:, 1:].T, (2.*np.pi/ks)**3)*4.*np.pi # Convert Delta^2(k) to P(k)

    # Loop over redshifts
    Pk_result = ks
    for z in zs:

        # Linear power
        Pks_lin = Pk_lin(z, ks)

        # Initialise halo model
        Omega_mz = Omega_m*(1.+z)**3/(Omega_m*(1.+z)**3.+(1.-Omega_m))
        dc = cosmology.dc_NakamuraSuto(Omega_mz)
        Dv = cosmology.Dv_BryanNorman(Omega_mz)
        hmod = halo.model(z, Omega_m, name=halomodel_name, Dv=Dv, dc=dc)
        print(hmod)

        # Halo profile
        Rs = hmod.Lagrangian_radius(Ms)
        sigmaRs = camb_results.get_sigmaR(Rs, hubble_units=True, return_R_z=False)[zs.index(z)]
        rvs = hmod.virial_radius(Ms)
        cs = halo.concentration(Ms, z, method=concentration_name, halo_definition=halo_definition)
        matter_profile = halo.matter_profile(ks, Ms, rvs, cs, hmod.Om_m)

        # Power spectrum calculation
        _, _, Pk = hmod.power_spectrum(ks, Pks_lin, Ms, sigmaRs, {'m': matter_profile})
        Pk_result = np.column_stack((Pk_result, Pk['m-m']))

    # Save data
    outfile = 'results/'+file
    np.savetxt(outfile, Pk_result)

        # Carry out test
        #np.testing.assert_array_almost_equal(Pk['m-m']/Pk_benchmark, 1., decimal=3)

### ###

class TestPower(unittest.TestCase):

    # Power-spectrum calculation
    @staticmethod
    def test_power():
        pass

### ###

### Unittest ###

if __name__ == '__main__':
    unittest.main()

### ###
