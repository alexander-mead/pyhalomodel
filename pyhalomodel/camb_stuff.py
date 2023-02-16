import camb

def run(zs, Omega_c, Omega_b, Omega_k, h, ns, sigma_8, 
    m_nu=0., w=-1., wa=0., As=2e-9, norm_sigma8=True, kmax_CAMB=200., zmax_CAMB=10., verbose=True):

    # Sets cosmological parameters in camb to calculate the linear power spectrum
    pars = camb.CAMBparams()
    wb, wc = Omega_b*h**2, Omega_c*h**2

    # This function sets standard and helium set using BBN consistency
    pars.set_cosmology(ombh2=wb, omch2=wc, H0=100.*h, mnu=m_nu, omk=Omega_k)
    pars.set_dark_energy(w=w, wa=wa, dark_energy_model='ppf') 
    pars.InitPower.set_params(As=As, ns=ns, r=0.)
    pars.set_matter_power(redshifts=zs, kmax=kmax_CAMB) # Setup the linear matter power spectrum
    Omega_m = pars.omegam # Extract the matter density

    # Scale 'As' to be correct for the desired 'sigma_8' value if necessary
    if norm_sigma8:
        results = camb.get_results(pars)
        sigma_8_init = (results.get_sigma8()[zs.index(0.)]).item()
        if verbose: 
            print('Running CAMB')
            print('Initial sigma_8:', sigma_8_init)
            print('Desired sigma_8:', sigma_8)
        scaling = (sigma_8/sigma_8_init)**2
        As *= scaling
        pars.InitPower.set_params(As=As, ns=ns, r=0.)

    # Run
    results = camb.get_results(pars)

    # Now get the linear power spectrum *interpolator* (note the .P)
    Pk = camb.get_matter_power_interpolator(pars, 
                                            nonlinear=False, 
                                            kmax=kmax_CAMB,
                                            zmax=zmax_CAMB,
                                            var1=camb.model.Transfer_tot,
                                            var2=camb.model.Transfer_tot, 
                                            ).P # Note the .P to get the interpolator
    sigma_8 = (results.get_sigma8()[zs.index(0.)]).item()
    if verbose:
        print('Final sigma_8:', sigma_8)
        print()

    # Return
    return Pk, results, Omega_m, sigma_8, As