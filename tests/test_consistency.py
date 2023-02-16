# Standard imports
import numpy as np
import unittest

# Project imports
import pyhalomodel.cosmology as cosmology
import pyhalomodel.camb_stuff as camb_stuff
import pyhalomodel as halo

### Parameters ###

# Set cosmological parameters
Omega_c = 0.25
Omega_b = 0.05
Omega_k = 0.0
h = 0.7
ns = 0.96
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
    'PS74': 'Press & Schecter (1974)',
    'ST99': 'Sheth & Tormen (1999)',
    'SMT01': 'Sheth, Mo & Tormen (2001)',
    'Tinker2010': 'Tinker et al. (2010)',
    'Despali2016': 'Despali et al. (2016)',
}

### ###

### CAMB ###

Pk_lin, camb_results, Omega_m, _, _ = camb_stuff.run(zs, Omega_c, Omega_b, Omega_k, h, ns, sigma_8)

# Loop over halo models
benchmarks_dict = {}; results_dict = {}
for short_name, halomodel_name in halomodel_names.items():

    # Loop over redshifts
    benchmarks = []; results = []
    for z in zs:

        # Loop over spectra type
        datas = {}
        for spectrum in ['mm', 'mg', 'gg']:
            file = 'power_'+spectrum+'_'+short_name+'_z%1.1f.dat'%(z)
            infile = 'benchmarks/'+file
            data = np.loadtxt(infile)
            ks = data[:, 0] # Wavenumbers
            datas[spectrum] = data[:, 1] # Power
        benchmark = np.column_stack((ks, datas['mm'], datas['mg'], datas['gg']))
        benchmarks.append(benchmark)
        ks = benchmark[:, 0]
        Pks_lin = Pk_lin(z, ks)

        # Initialise halo model
        Omega_mz = Omega_m*(1.+z)**3/(Omega_m*(1.+z)**3.+(1.-Omega_m))
        dc = cosmology.dc_NakamuraSuto(Omega_mz)
        Dv = cosmology.Dv_BryanNorman(Omega_mz)
        hmod = halo.model(z, Omega_m, name=halomodel_name, Dv=Dv, dc=dc)

        # Halo properties
        Rs = hmod.Lagrangian_radius(Ms)
        sigmaRs = camb_results.get_sigmaR(Rs, hubble_units=True, return_R_z=False)[zs.index(z)]
        rvs = hmod.virial_radius(Ms)
        cs = halo.concentration(Ms, z, halo_definition=halo_definition)

        # Create a matter profile
        matter_profile = halo.matter_profile(ks, Ms, rvs, cs, hmod.Om_m)

        # Create galaxy profiles
        N_cen, N_sat = halo.HOD_mean(Ms, method='Zheng et al. (2005)')
        V_cen, V_sat, _ = halo.HOD_variance(N_cen, N_sat)
        N_gal = N_cen+N_sat; V_gal = V_cen+V_sat
        rho_gal = hmod.average(Ms, sigmaRs, N_cen+N_gal)
        Uk_gal = halo.window_function(ks, rvs, profile='isothermal')
        galaxy_profile = halo.profile.Fourier(ks, Ms, Uk_gal, amplitude=N_gal, normalisation=rho_gal, variance=V_gal, discrete_tracer=True)

        # Power spectrum calculation
        _, _, Pk = hmod.power_spectrum(ks, Pks_lin, Ms, sigmaRs, {'m': matter_profile, 'g': galaxy_profile})

        # Save data
        stuff = []
        for thing1, thing2 in zip(['mm', 'mg', 'gg'], ['m-m', 'm-g', 'g-g']):
            file = 'power_'+thing1+'_'+short_name+'_z%1.1f.dat'%(z)
            outfile = 'results/'+file
            result = np.column_stack((ks, Pk[thing2]))
            np.savetxt(outfile, result, header='k [h/Mpc]; P_'+thing1+'(k) [(Mpc/h)^3]')
            stuff.append(Pk[thing2])
        result = np.column_stack((ks, *stuff))
        results.append(result)

    # Add to dictionaries
    benchmarks_dict[halomodel_name] = benchmarks
    results_dict[halomodel_name] = results

### ###

class TestPower(unittest.TestCase):

    # matter-matter

    # PS '74
    @staticmethod
    def test_mm_PS74():
        name = 'Press & Schecter (1974)'; n = 1
        for benchmark, result in zip(benchmarks_dict[name], results_dict[name]):
            np.testing.assert_array_almost_equal(result[:, n]/benchmark[:, n], 1., decimal=4)

    # ST '99
    @staticmethod
    def test_mm_ST99():
        name = 'Sheth & Tormen (1999)'; n = 1
        for benchmark, result in zip(benchmarks_dict[name], results_dict[name]):
            np.testing.assert_array_almost_equal(result[:, n]/benchmark[:, n], 1., decimal=4)

    # SMT '01
    @staticmethod
    def test_mm_SMT01():
        name = 'Sheth, Mo & Tormen (2001)'; n = 1
        for benchmark, result in zip(benchmarks_dict[name], results_dict[name]):
            np.testing.assert_array_almost_equal(result[:, n]/benchmark[:, n], 1., decimal=4)

    # Tinker '10
    @staticmethod
    def test_mm_Tinker2010():
        name = 'Tinker et al. (2010)'; n = 1
        for benchmark, result in zip(benchmarks_dict[name], results_dict[name]):
            np.testing.assert_array_almost_equal(result[:, n]/benchmark[:, n], 1., decimal=4)

    # Despali '16
    @staticmethod
    def test_mm_Despali2016():
        name = 'Despali et al. (2016)'; n = 1
        for benchmark, result in zip(benchmarks_dict[name], results_dict[name]):
            np.testing.assert_array_almost_equal(result[:, n]/benchmark[:, n], 1., decimal=4)

    # matter-galaxy

    # PS '74
    @staticmethod
    def test_mg_PS74():
        name = 'Press & Schecter (1974)'; n = 2
        for benchmark, result in zip(benchmarks_dict[name], results_dict[name]):
            np.testing.assert_array_almost_equal(result[:, n]/benchmark[:, n], 1., decimal=4)

    # ST '99
    @staticmethod
    def test_mg_ST99():
        name = 'Sheth & Tormen (1999)'; n = 2
        for benchmark, result in zip(benchmarks_dict[name], results_dict[name]):
            np.testing.assert_array_almost_equal(result[:, n]/benchmark[:, n], 1., decimal=4)

    # SMT '01
    @staticmethod
    def test_mg_SMT01():
        name = 'Sheth, Mo & Tormen (2001)'; n = 2
        for benchmark, result in zip(benchmarks_dict[name], results_dict[name]):
            np.testing.assert_array_almost_equal(result[:, n]/benchmark[:, n], 1., decimal=4)

    # Tinker '10
    @staticmethod
    def test_mg_Tinker2010():
        name = 'Tinker et al. (2010)'; n = 2
        for benchmark, result in zip(benchmarks_dict[name], results_dict[name]):
            np.testing.assert_array_almost_equal(result[:, n]/benchmark[:, n], 1., decimal=4)

    # Despali '16
    @staticmethod
    def test_mg_Despali2016():
        name = 'Despali et al. (2016)'; n = 2
        for benchmark, result in zip(benchmarks_dict[name], results_dict[name]):
            np.testing.assert_array_almost_equal(result[:, n]/benchmark[:, n], 1., decimal=4)

    # galaxy-galaxy

    # PS '74
    @staticmethod
    def test_gg_PS74():
        name = 'Press & Schecter (1974)'; n = 3
        for benchmark, result in zip(benchmarks_dict[name], results_dict[name]):
            np.testing.assert_array_almost_equal(result[:, n]/benchmark[:, n], 1., decimal=4)

    # ST '99
    @staticmethod
    def test_gg_ST99():
        name = 'Sheth & Tormen (1999)'; n = 3
        for benchmark, result in zip(benchmarks_dict[name], results_dict[name]):
            np.testing.assert_array_almost_equal(result[:, n]/benchmark[:, n], 1., decimal=4)

    # SMT '01
    @staticmethod
    def test_gg_SMT01():
        name = 'Sheth, Mo & Tormen (2001)'; n = 3
        for benchmark, result in zip(benchmarks_dict[name], results_dict[name]):
            np.testing.assert_array_almost_equal(result[:, n]/benchmark[:, n], 1., decimal=4)

    # Tinker '10
    @staticmethod
    def test_gg_Tinker2010():
        name = 'Tinker et al. (2010)'; n = 3
        for benchmark, result in zip(benchmarks_dict[name], results_dict[name]):
            np.testing.assert_array_almost_equal(result[:, n]/benchmark[:, n], 1., decimal=4)

    # Despali '16
    @staticmethod
    def test_gg_Despali2016():
        name = 'Despali et al. (2016)'; n = 3
        for benchmark, result in zip(benchmarks_dict[name], results_dict[name]):
            np.testing.assert_array_almost_equal(result[:, n]/benchmark[:, n], 1., decimal=4)

### ###

### Unittest ###

if __name__ == '__main__':
    unittest.main()

### ###
