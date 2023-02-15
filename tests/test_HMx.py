# Standard imports
import numpy as np
import unittest

# Projet imports
import pyhalomodel.common.cosmology as cosmology
import pyhalomodel.common.camb_stuff as camb_stuff
import pyhalomodel as halo

### Parameters ###

# Set cosmological parameters
Omega_c = 0.25
Omega_b = 0.05
Omega_k = 0.
h = 0.7
ns = 0.96
sigma_8_set = True # If True uses the following value
sigma_8  = 0.8

# Redshifts
zs = [3., 2., 1., 0.5, 0.]

# Halo model
Mmin, Mmax = 1e9, 1e17
nM = 256
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

### Generate results ###

Pk_lin, camb_results, Omega_m, _, _ = camb_stuff.run(zs, Omega_c, Omega_b, Omega_k, h, ns, sigma_8)

# Loop over halo models
benchmarks_dict, results_dict = {}, {}
for halomodel_name, halomodel_number in halomodel_names.items():

    # Read benchmark
    file = 'HMx_26_'+str(halomodel_number)+'.dat'
    infile = 'benchmarks/'+file
    benchmark = np.loadtxt(infile)
    ks = benchmark[:, 0] # Wavenumbers
    Pk_benchmark = benchmark[:, 1:].T*(2.*np.pi/ks)**3/(4.*np.pi) # Convert Delta^2(k) to P(k)
    benchmarks_dict[halomodel_name] = np.column_stack((ks, Pk_benchmark.T))

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
    results_dict[halomodel_name] = Pk_result
    outfile = 'results/'+file
    np.savetxt(outfile, Pk_result)

### ###

### Tests ###

class TestPower(unittest.TestCase):

    @staticmethod
    def test_PS74():
        name = 'Press & Schecter (1974)'
        for benchmark, result in zip(benchmarks_dict[name], results_dict[name]):
            np.testing.assert_array_almost_equal(result/benchmark, 1., decimal=2)

    @staticmethod
    def test_ST99():
        name = 'Sheth & Tormen (1999)'
        for benchmark, result in zip(benchmarks_dict[name], results_dict[name]):
            np.testing.assert_array_almost_equal(result/benchmark, 1., decimal=2)

    @staticmethod
    def test_SMT01():
        name = 'Sheth, Mo & Tormen (2001)'
        for benchmark, result in zip(benchmarks_dict[name], results_dict[name]):
            np.testing.assert_array_almost_equal(result/benchmark, 1., decimal=2)

    @staticmethod
    def test_Tinker2010():
        name = 'Tinker et al. (2010)'
        for benchmark, result in zip(benchmarks_dict[name], results_dict[name]):
            np.testing.assert_array_almost_equal(result/benchmark, 1., decimal=2)

    @staticmethod
    def test_Despali2016():
        name = 'Despali et al. (2016)'
        for benchmark, result in zip(benchmarks_dict[name], results_dict[name]):
            np.testing.assert_array_almost_equal(result/benchmark, 1., decimal=2)

### ###

### Unittest ###

if __name__ == '__main__':
    unittest.main()

### ###
