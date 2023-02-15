# Standard imports
import numpy as np

# Third-party imports
import camb

# Project imports
import pyhalomodel.cosmology as cosmology
import pyhalomodel as halo

### Parameters ###

# Cosmological parameters
Omega_c = 0.25
Omega_b = 0.05
Omega_k = 0.
h = 0.7
As = 2e-9
ns = 0.96
w = -1.
wa = 0.
m_nu = 0. # [eV]
sigma_8_set = True
sigma_8  = 0.8

# Wavenumber range [h/Mpc]
kmin = 1e-3; kmax = 1e1
nk = 101
ks = np.logspace(np.log10(kmin), np.log10(kmax), nk)

# Redshifts
zs = [3., 2., 1., 0.5, 0.]

# CAMB
kmax_CAMB = 200. # Maximum wavenumber [h/Mpc]; should be larger than you actually want
zmax_CAMB = 10.  # Maximum redshift; should be larger than you actually want

# Halo model
Mmin = 1e9; Mmax = 1e17
nM = 256
Ms = np.logspace(np.log10(Mmin), np.log10(Mmax), nM)
halo_definition = 'Mvir'
concentration_name = 'Duffy et al. (2008)'
HOD_name = 'Zheng et al. (2005)'

halomodel_names = {
    'PS74': 'Press & Schecter (1974)',
    'ST99': 'Sheth & Tormen (1999)',
    'SMT01': 'Sheth, Mo & Tormen (2001)',
    'Tinker2010': 'Tinker et al. (2010)',
    'Despali2016': 'Despali et al. (2016)',
}

### ###

### CAMB ###

# Sets cosmological parameters in camb to calculate the linear power spectrum
pars = camb.CAMBparams()
wb, wc = Omega_b*h**2, Omega_c*h**2

# This function sets standard and helium set using BBN consistency
pars.set_cosmology(ombh2=wb, omch2=wc, H0=100.*h, mnu=m_nu, omk=Omega_k)
pars.set_dark_energy(w=w, wa=wa, dark_energy_model='ppf') 
pars.InitPower.set_params(As=As, ns=ns, r=0)
pars.set_matter_power(redshifts=zs, kmax=kmax_CAMB) # Setup the linear matter power spectrum

# Scale 'As' to be correct for the desired 'sigma_8' value if necessary
if sigma_8_set:
    camb_results = camb.get_results(pars)
    sigma_8_init = (camb_results.get_sigma8()[zs.index(0.)]).item()
    scaling = (sigma_8/sigma_8_init)**2
    As *= scaling
    pars.InitPower.set_params(As=As, ns=ns, r=0)

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
sigma_8 = (camb_results.get_sigma8()[zs.index(0.)]).item()

### ###

### Halo model ###

# Halo-model names
for short_name, halomodel_name in halomodel_names.items():

    # Loop over redshifts
    for z in zs:

        # Initialise halo model
        Om_mz = Omega_m*(1.+z)**3/(Omega_m*(1.+z)**3.+(1.-Omega_m))
        dc = cosmology.dc_NakamuraSuto(Om_mz)
        Dv = cosmology.Dv_BryanNorman(Om_mz)
        hmod = halo.model(z, Omega_m, name=halomodel_name, Dv=Dv, dc=dc)

        # Linear power
        Pks_lin = Pk_lin(z, ks)

        # Halo properties
        Rs = hmod.Lagrangian_radius(Ms)
        sigmaRs = camb_results.get_sigmaR(Rs, hubble_units=True, return_R_z=False)[zs.index(z)]
        rvs = hmod.virial_radius(Ms)
        cs = halo.concentration(Ms, z, method='Duffy et al. (2008)', halo_definition=halo_definition)

        # Create matter profiles
        matter_profile = halo.matter_profile(ks, Ms, rvs, cs, hmod.Om_m)

        # Create galaxy profiles
        N_cen, N_sat = halo.HOD_mean(Ms, method='Zheng et al. (2005)')
        V_cen, V_sat, _ = halo.HOD_variance(N_cen, N_sat)
        N_gal = N_cen+N_sat; V_gal = V_cen+V_sat
        rho_gal = hmod.average(Ms, sigmaRs, N_cen+N_gal)
        Uk_gal = halo.window_function(ks, rvs, profile='isothermal')
        galaxy_profile = halo.profile.Fourier(ks, Ms, Uk_gal, amp=N_gal, norm=rho_gal, var=V_gal, discrete_tracer=True)

        # Power-spectrum calculation
        _, _, Pk = hmod.power_spectrum(ks, Pks_lin, Ms, sigmaRs, {'m': matter_profile, 'g': galaxy_profile})

        # Save results (matter-matter)
        data = np.column_stack((ks, Pk['m-m']))
        outfile = 'benchmarks/power_mm_'+short_name+'_z%1.1f.dat'%(z)
        with open(outfile, 'x') as f:
            np.savetxt(f, data, header='k [h/Mpc]; P_mm(k) [(Mpc/h)^3]')

        # Save results (matter-galaxy)
        data = np.column_stack((ks, Pk['m-g']))
        outfile = 'benchmarks/power_mg_'+short_name+'_z%1.1f.dat'%(z)
        with open(outfile, 'x') as f:
            np.savetxt(f, data, header='k [h/Mpc]; P_mg(k) [(Mpc/h)^3]')

        # Save results (galaxy-galaxy)
        data = np.column_stack((ks, Pk['g-g']))
        outfile = 'benchmarks/power_gg_'+short_name+'_z%1.1f.dat'%(z)
        with open(outfile, 'x') as f:
            np.savetxt(f, data, header='k [h/Mpc]; P_gg(k) [(Mpc/h)^3]')

### ###
