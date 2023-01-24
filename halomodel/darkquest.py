# Standard imports
import numpy as np
import warnings

# Other imports
from dark_emulator import darkemu

# My imports
import constants as const
import utility as util

# Constants
dc = 1.686      # Collapse threshold for nu definition
Dv = 200.       # Spherical-overdensity halo definition
np_min = 200    # Minimum number of halo particles
npart = 2048    # Cube root of number of simulation particles
Lbox_HR = 1000. # Box size for high-resolution simulations [Mpc/h]
Lbox_LR = 2000. # Box size for low-resolution simulations [Mpc/h]

# Maximum redshift
zmax = 1.48

# Minimum and maximum values of cosmological parameters in the emulator
wb_min = 0.0211375
wb_max = 0.0233625
wc_min = 0.10782
wc_max = 0.13178
Om_w_min = 0.54752
Om_w_max = 0.82128
lnAs_min = 2.4752
lnAs_max = 3.7128
ns_min = 0.916275
ns_max = 1.012725
w_min = -1.2
w_max = -0.8

# Fiducial cosmology
wb_fid = 0.02225
wc_fid = 0.1198
Om_w_fid = 0.6844
lnAs_fid = 3.094
ns_fid = 0.9645
w_fid = -1.

# Parameters
# TODO: Setting this to True messes things up, fix it!
log_interp_sigma = False

# Accuracy
acc_hh = 0.04
acc_hm = 0.02

# Distance from low/high boundary when varying cosmology along a parameter-cube direction
low_fac = 0.15 
high_fac = 0.85

## Beta-NL ##

# Source of linear halo bias
# 1 - From emulator 'bias' function
# 2 - From emulator halo-halo spectrum at large wavenumber
# 3 - From emulator halo-matter spectrum at large wavenumber
ibias_BNL = 2

# Force to zero at large scales?

# Large 'linear' scale [h/Mpc]
klin_BNL = 0.02

## ##

class cosmology():

    # My version of a Dark Quest cosmology class

    def __init__(self, wb=wb_fid, wc=wc_fid, Om_w=Om_w_fid, lnAs=lnAs_fid, ns=ns_fid, w=w_fid):

        # Primary parameters
        self.wb = wb
        self.wc = wc
        self.Om_w = Om_w
        self.lnAs = lnAs
        self.ns = ns
        self.w = w

        # Fixed parameters
        self.wnu = 0.00064
        self.Om_k = 0.

        # Derived parameters
        self.wm = self.wc+self.wb+self.wnu
        self.Om_m = 1.-self.Om_w # Flatness
        self.h = np.sqrt(self.wm/self.Om_m)
        self.Om_b = self.wb/self.h**2
        self.Om_c = self.wc/self.h**2
        self.As = np.exp(self.lnAs)/1e10
        self.m_nu = self.wnu*const.nuconst
        self.Om_nu = self.wnu/self.h**2

    def cosmology_array(self):
        return np.atleast_2d([self.wb, self.wc, self.Om_w, self.lnAs, self.ns, self.w])

    def print(self):

        # Write primary parameters to screen
        print('Dark Quest primary parameters')
        print('omega_b: %1.4f' % (self.wb))
        print('omega_c: %1.4f' % (self.wc))  
        print('Omega_w: %1.4f' % (self.Om_w))
        print('As [1e9]: %1.4f' % (self.As*1e9))
        print('ns: %1.4f' % (self.ns))
        print('w: %1.4f' % (self.w))
        print()

        #print('Dark Quest fixed parameters')
        #print('omega_nu: %1.4f' % (self.wnu))
        #print('Omega_k: %1.4f' % (self.Om_k))
        #print()

        # Write derived parameters to screen
        print('Dark Quest derived parameters')
        print('Omega_m: %1.4f' % (self.Om_m))
        print('Omega_b: %1.4f' % (self.Om_b))      
        print('omega_m: %1.4f' % (self.wm))
        print('h: %1.4f' % (self.h))      
        print('Omega_c: %1.4f' % (self.Om_c))
        print('Omega_nu: %1.4f' % (self.Om_nu))      
        print('m_nu [eV]: %1.4f' % (self.m_nu))
        print()


def random_cosmology():
    '''
    (Uniform) random cosmological parameters from within the Dark Quest hypercube
    '''
    wb = np.random.uniform(wb_min, wb_max)
    wc = np.random.uniform(wc_min, wc_max)
    Om_w = np.random.uniform(Om_w_min, Om_w_max)
    lnAs = np.random.uniform(lnAs_min, lnAs_max)
    ns = np.random.uniform(ns_min, ns_max)
    w = np.random.uniform(w_min, w_max)

    cpar = cosmology(wb=wb, wc=wc, Om_w=Om_w, lnAs=lnAs, ns=ns, w=w)
    return cpar


def named_cosmology(name):

    # Start from fiducial cosmology
    wb = wb_fid
    wc = wc_fid
    Om_w = Om_w_fid
    lnAs = lnAs_fid
    ns = ns_fid
    w = w_fid 

    # Vary some parameters
    if name in ['low w_b', 'low w_c', 'low Om_w', 'low lnAs', 'low ns', 'low w', 
        'high w_b', 'high w_c', 'high Om_w', 'high lnAs', 'high ns', 'high w']:
        if name in ['low w_b', 'low w_c', 'low Om_w', 'low lnAs', 'low ns', 'low w']:
            fac = low_fac
        elif name in ['high w_b', 'high w_c', 'high Om_w', 'high lnAs', 'high ns', 'high w']:
            fac = high_fac
        else:
            raise ValueError('Cosmology name not recognised')
        if name in ['low w_b', 'high w_b']:
            wb = wb_min+(wb_max-wb_min)*fac
        elif name in ['low w_c', 'high w_c']:
            wc = wc_min+(wc_max-wc_min)*fac
        elif name in ['low Om_w', 'high Om_w']:
            Om_w = Om_w_min+(Om_w_max-Om_w_min)*fac
        elif name in ['low lnAs', 'high lnAs']:
            lnAs = lnAs_min+(lnAs_max-lnAs_min)*fac
        elif name in ['low ns', 'high ns']:
            ns = ns_min+(ns_max-ns_min)*fac
        elif name in ['low w', 'high w']:
            w = w_min+(w_max-w_min)*fac
        else:
            raise ValueError('Cosmology name not recognised')
    elif name == 'Multidark':
            wb = 0.0230
            wc = 0.1093
            Om_w = 0.73
            lnAs = 3.195
            ns = 0.95
            w = -1.
    else:
        raise ValueError('Cosmology name not recognised')

    # Create my version of the Dark Quest cosmology object
    cpar = cosmology(wb=wb, wc=wc, Om_w=Om_w, lnAs=lnAs, ns=ns, w=w)
    return cpar


def init_emulator(cpar):
    '''
    Initialise the emulator for a given set of cosmological parameters
    cpar: My verion of Dark Quest cosmology object
    '''
    # Start Dark Quest
    print('Initialize Dark Quest')
    emu = darkemu.base_class()
    print('')

    # Initialise emulator
    #cparam = np.array([cpar.wb, cpar.wc, cpar.Om_w, cpar.lnAs, cpar.ns, cpar.w]) # Surely this should be a dictionary
    cpar.print()
    cparam = cpar.cosmology_array()#np.array(cpar.list())
    emu.set_cosmology(cparam) # This does a load of emulator init steps
    cpar.sig8 = emu.get_sigma8()
    print('Derived sigma_8:', cpar.sig8)
    print()

    return emu


def get_Pk_mm(emu, ks, zs, nonlinear=False):
    '''
    Matter power spectrum from emulator; either linear or non-linear
    '''
    if isinstance(zs, float):
        if nonlinear:
            Pk = emu.get_pknl(ks, zs)
        else:         
            Pk = emu.get_pklin_from_z(ks, zs)
    else:
        Pk = np.zeros((len(zs), len(ks)))
        for iz, z in enumerate(zs):
            if nonlinear:
                Pk[iz, :] = emu.get_pknl(ks, z)
            else:         
                Pk[iz, :] = emu.get_pklin_from_z(ks, z)
    return Pk


def minimum_halo_mass(emu):
    '''
    Minimum halo mass for the set of cosmological parameters [Msun/h]
    '''
    Mbox_HR = comoving_matter_density(emu)*Lbox_HR**3
    mmin = Mbox_HR*np_min/npart**3
    return mmin


def comoving_matter_density(emu):
    '''
    Comoving matter density [(Msun/h)/(Mpc/h)^3]
    '''
    Om_m = emu.cosmo.get_Omega0()
    rhom = cosmology.comoving_matter_density(Om_m)
    return rhom


def nu_R(emu, R, z):
    '''
    nu = dc/sigma [dimensionless]
    '''
    M = mass_R(emu, R)
    return nu_M(emu, M, z)


def nu_M(emu, M, z):
    '''
    nu = dc/sigma [dimensionless]
    '''
    return dc/sigma_M(emu)(M, z)


def Lagrangian_radius(emu, M):
    '''
    Lagrangian radius (comoving) of a halo of mass M [Mpc/h]
    '''
    Om_m = emu.cosmo.get_Omega0()
    return cosmology.Lagrangian_radius(M, Om_m)


def virial_radius(emu, M):
    '''
    Virial radius (comoving) of a halo of mass M [Mpc/h]
    '''
    return Lagrangian_radius(emu, M)/np.cbrt(Dv)


def mass_R(emu, R):
    '''
    Mass enclosed within comoving radius R [Msun/h]
    '''
    Om_m = emu.cosmo.get_Omega0()
    return cosmology.Mass_R(R, Om_m)


def mass_nu(emu, nu, z):

    # TODO: This does both interpolation and evaluation, could/should split up?

    # Import
    from scipy.interpolate import InterpolatedUnivariateSpline as ius

    # Options
    log_interp = log_interp_sigma # Should sigma(M) be interpolated logarithmically?
    
    # Get internal M vs sigma arrays
    Ms_internal = emu.massfunc.Mlist
    sig0s_internal = emu.massfunc.sigs0
    sigs_internal = sig0s_internal*emu.Dgrowth_from_z(z)
    nus_internal = dc/sigs_internal 

    # Make an interpolator for sigma(M)  
    if log_interp:
        mass_interpolator = ius(nus_internal, np.log(Ms_internal))
    else:
        mass_interpolator = ius(nus_internal, Ms_internal)

    # Get sigma(M) from the interpolator at the desired masses
    if log_interp:
        Mass = np.exp(mass_interpolator(nu))
    else:
        Mass = mass_interpolator(nu)

    return Mass


def non_linear_mass(emu, z):
    '''
    Returns non-linear mass, M* | nu(M*) = 1 [Msun/h]
    '''
    return  mass_nu(emu, 1., z)

# def sigma_M(emu, M, z):

#     # TODO: This creates the interpolator AND evaluates it. Could just create an interpolator...

#     # Import
#     from scipy.interpolate import InterpolatedUnivariateSpline as ius

#     # Options
#     log_interp = log_interp_sigma # Should sigma(M) be interpolated logarithmically?
    
#     # Get internal M vs sigma arrays
#     Ms_internal = emu.massfunc.Mlist
#     sigs_internal = emu.massfunc.sigs0

#     # Make an interpolator for sigma(M)  
#     if log_interp:
#         sigma_interpolator = ius(np.log(Ms_internal), np.log(sigs_internal), ext='extrapolate')
#     else:
#         sigma_interpolator = ius(Ms_internal, sigs_internal, ext='extrapolate')
    
#     # Get sigma(M) from the interpolator at the desired masses
#     if log_interp:
#         sigma0 = np.exp(sigma_interpolator(np.log(M)))
#     else:
#         sigma0 = sigma_interpolator(M)

#     # Growth function (g(z=0)=1)
#     g = emu.Dgrowth_from_z(z)
#     sigma = g*sigma0

#     # Result assuming scale-independent growth
#     return sigma

def sigma_M(emu):
    '''
    Create an interpolator for sigma(M)
    TODO: Attach this to emu class?
    '''
    from scipy.interpolate import InterpolatedUnivariateSpline as ius
    log_interp = log_interp_sigma # Should sigma(M) be interpolated logarithmically?
    
    # Get internal M vs sigma arrays
    Ms_internal = emu.massfunc.Mlist
    sigs_internal = emu.massfunc.sigs0
    g = emu.Dgrowth_from_z # This is a function

    # Make an interpolator for sigma(M)  
    if log_interp:
        sigma_interpolator = ius(np.log(Ms_internal), np.log(sigs_internal), ext='extrapolate')
    else:
        sigma_interpolator = ius(Ms_internal, sigs_internal, ext='extrapolate')
    return lambda M, z: g(z)*sigma_interpolator(M)


def sigma_R(emu, R, z):
    '''
    Root-mean-square linear overdensity fluctuation when field smoothed on scale R [dimensionless]
    args:
        emu: An instance of DQ emulator
        R: Radius [Mpc/h] (TODO: can this be a list?)
        z: redshift
    '''
    M = mass_R(emu, R)
    return sigma_M(emu)(M, z)

def get_sigma_Ms(emu, Ms, z):
    '''
    Returns an array of sigma(Ms, z) values 
    '''
    log_interp = log_interp_sigma
    if log_interp:
        sigmas = sigma_M(emu)(np.log(Ms), z)
    else:
        sigmas = sigma_M(emu)(Ms, z)
    return sigmas


def get_sigma_Rs(emu, Rs, z):
    '''
    Returns an array of sigma(Rs, z) values 
    '''
    Ms = mass_R(emu, Rs)
    return get_sigma_Ms(emu, Ms, z)


def get_bias_mass(emu, M, redshift):
    '''
    Linear halo bias: b(M, z)
    Taken from the pre-release version of Dark Quest given to me by Takahiro
    I am not sure why this functionality was omitted from the final version
    '''
    Mp = M * 1.01
    Mm = M * 0.99
    logdensp = np.log10(emu.mass_to_dens(Mp, redshift))
    logdensm = np.log10(emu.mass_to_dens(Mm, redshift))
    bp = emu.get_bias(logdensp, redshift)
    bm = emu.get_bias(logdensm, redshift)
    return (bm * 10**logdensm - bp * 10**logdensp) / (10**logdensm - 10**logdensp)


def get_dndM_mass(emu, Ms, z):
    '''
    Return an array of n(M) (dn/dM in Dark Quest notation) at user-specified halo masses
    Extrapolates if necessary, which is perhaps dangerous
    '''
    from scipy.interpolate import InterpolatedUnivariateSpline as ius

    # Construct an interpolator for n(M) (or dn/dM) from the emulator internals
    Ms = emu.massfunc.Mlist
    dndM = emu.massfunc.get_dndM(z)
    dndM_interp = ius(np.log(Ms), np.log(dndM), ext='extrapolate')

    # Evaluate the interpolator at the desired mass points
    return np.exp(dndM_interp(np.log(Ms)))


def ndenshalo(emu, Mmin, Mmax, z):
    '''
    Calculate the number density of haloes in the range Mmin to Mmax
    Result is [(Mpc/h)^-3]
    '''
    vol = 1. # Fix to unity
    return emu.get_nhalo(Mmin, Mmax, vol, z)


def mass_avg(emu, Mmin, Mmax, z, pow=1):
    '''
    Calculate the average halo mass between two limits, weighted by the halo mass function [Msun/h]
    '''
    from scipy.interpolate import InterpolatedUnivariateSpline as ius
    from scipy.integrate import quad

    # Parameters
    epsabs = 1e-5 # Integration accuracy

    # Construct an interpolator for n(M) (or dn/dM) from the emulator internals
    Ms = emu.massfunc.Mlist
    dndM = emu.massfunc.get_dndM(z)
    log_dndM_interp = ius(np.log(Ms), np.log(dndM), ext='extrapolate')

    # Number density of haloes in the mass range
    n = ndenshalo(emu, Mmin, Mmax, z) 

    # Integrate to get the average mass
    Mav, _ = quad(lambda M: (M**pow)*np.exp(log_dndM_interp(np.log(M))), Mmin, Mmax, epsabs=epsabs)

    return Mav/n


def get_xiauto_mass_avg(emu, rs, M1min, M1max, M2min, M2max, z):
    '''
    Averages the halo-halo correlation function over mass ranges to return the weighted-by-mass-function mean version
    '''
    from scipy.interpolate import InterpolatedUnivariateSpline as ius
    from scipy.interpolate import RectBivariateSpline as rbs
    from scipy.integrate import dblquad

    # Parameters
    epsabs = 1e-3 # Integration accuracy
    nM = 6        # Number of halo-mass bins in each of M1 and M2 directions

    # Calculations
    nr = len(rs)

    # Number densities of haloes in each sample
    n1 = ndenshalo(emu, M1min, M1max, z)
    n2 = ndenshalo(emu, M2min, M2max, z)

    # Arrays for halo masses
    M1s = util.logspace(M1min, M1max, nM)
    M2s = util.logspace(M2min, M2max, nM)
    
    # Get mass function interpolation
    Ms = emu.massfunc.Mlist
    dndM = emu.massfunc.get_dndM(z)
    log_dndM_interp = ius(np.log(Ms), np.log(dndM))

    # Loop over radii
    xiauto_avg = np.zeros((nr))
    for ir, r in enumerate(rs):

        # Get correlation function interpolation
        # Note that this is not necessarily symmetric because M1, M2 run over different ranges
        xiauto_mass = np.zeros((nM, nM))
        for iM1, M1 in enumerate(M1s):
            for iM2, M2 in enumerate(M2s):
                xiauto_mass[iM1, iM2] = emu.get_xiauto_mass(r, M1, M2, z)
        xiauto_interp = rbs(np.log(M1s), np.log(M2s), xiauto_mass)

        # Integrate interpolated functions
        # TODO: Unused variables in lambda here
        func = lambda M1, M2: xiauto_interp(np.log(M1),np.log(M2))*np.exp(log_dndM_interp(np.log(M1))+log_dndM_interp(np.log(M2)))
        xiauto_avg[ir], _ = dblquad(func, M1min, M1max, lambda M1: M2min, lambda M1: M2max, epsabs=epsabs)

    return xiauto_avg/(n1*n2)


def get_linear_halo_bias(emu, M, z, klin, Pk_klin):
    '''
    Linear halo bias
    '''
    ibias = ibias_BNL # Source of linear halo bias
    if ibias == 1:
        b = get_bias_mass(emu, M, z)[0]
    elif ibias == 2:
        b = np.sqrt(emu.get_phh_mass(klin, M, M, z)/Pk_klin)
    elif ibias == 3:
        b = emu.get_phm_mass(klin, M, z)/Pk_klin
    else:
        raise ValueError('Linear bias recipe not recognised')
    return b


def R_hh(emu, ks, M1, M2, z):
    '''
    Cross correlation coefficient between halo masses
    TODO: Prefix with get_ ?
    '''
    P12 = emu.get_phh_mass(ks, M1, M2, z)
    P11 = emu.get_phh_mass(ks, M1, M1, z)
    P22 = emu.get_phh_mass(ks, M2, M2, z)
    if (P11<0).any() or (P22<0).any():
        warnings.warn('Negative values in halo auto power, M1=M2='+'%0.2f'%(np.log10(M1)), RuntimeWarning)
    return P12/np.sqrt(P11*P22)

### ###

### Non-linear halo bias ###

def get_beta_NL(emu, mass, ks, z, force_to_zero=0, mass_variable='Mass', knl=5.):
    '''
    Beta_NL function, function: B^NL(M1, M2, k)
    TODO: Change to accept two separate mass arguments and merge with beta_NL_1D?
    '''
    # Parameters
    klin = np.array([klin_BNL]) # klin must be a numpy array
    
    # Set array name sensibly
    if mass_variable == 'Mass':
        Ms = mass
    elif mass_variable == 'Radius':
        Rs = mass
        Ms = mass_R(emu, Rs)
    elif mass_variable == 'nu':
        nus = mass
        Ms = mass_nu(emu, nus, z)
    else:
        raise ValueError('Error, mass variable for beta_NL not recognised')
    
    # Linear power
    Pk_lin = emu.get_pklin_from_z(ks, z)
    Pk_klin = emu.get_pklin_from_z(klin, z)
    index_klin, _ = util.findClosestIndex(klin, ks)
    #index_knl, knl_closest = util.findClosestIndex(knl, ks)
    
    # Calculate beta_NL by looping over mass arrays
    beta = np.zeros((len(Ms), len(Ms), len(ks)))
    for iM1, M1 in enumerate(Ms):

        # Linear halo bias
        b1 = get_linear_halo_bias(emu, M1, z, klin, Pk_klin)

        for iM2, M2 in enumerate(Ms):

            if iM2 >= iM1:

                # Create beta_NL
                b2 = get_linear_halo_bias(emu, M2, z, klin, Pk_klin)
                Pk_hh = emu.get_phh_mass(ks, M1, M2, z)
                beta[iM1, iM2, :] = Pk_hh/(b1*b2*Pk_lin)-1.

                # Force Beta_NL to be zero at large scales if necessary
                if force_to_zero != 0:
                    Pk_hh0 = emu.get_phh_mass(klin, M1, M2, z)
                    db = Pk_hh0/(b1*b2*Pk_klin)-1.
                    if force_to_zero == 1:
                        beta[iM1, iM2, :] = beta[iM1, iM2, :]-db # Additive correction
                    elif force_to_zero == 2:
                        beta[iM1, iM2, :] = (beta[iM1, iM2, :]+1.)/(db+1.)-1. # Multiplicative correction
                    elif force_to_zero == 3:
                        # print('setting beta_nl=beta_nl(k='+'%0.3f' % klin_closest+') for k<'+ '%0.3f' % klin_closest)
                        beta[iM1, iM2, :index_klin] = beta[iM1, iM2, index_klin] # for k<klin use the value for klin or the closest to that
                    elif force_to_zero == 4:
                        # print('setting beta_nl=0 for k<'+ '%0.3f' % klin_closest)
                        beta[iM1, iM2, :index_klin] = 0.0 # for k<klin set all values to zero
                        # beta[iM1, iM2, index_knl:] = 0.0 
                    elif force_to_zero == 5:
                        beta[iM1, iM2, :] = beta[iM1, iM2, :]*(1.-np.exp(-(ks/klin)**2.)) #Smoothly go to zero at k~klin
                    elif force_to_zero == 6:
                        # like 5 but with dependence on M1 and M2
                        if (iM1 == iM2) or (np.log10(M2/M1)<1):
                            truncate_bnl= (1.-np.exp(-(ks/klin)))
                        else:
                            truncate_bnl= (1.-np.exp(-(ks/klin)*np.log10(M2/M1)))
                        beta[iM1, iM2, :] = beta[iM1, iM2, :]*truncate_bnl #Change to smoothly going to zero
                    else:
                        raise ValueError('force_to_zero not set correctly, choose from 0-6')

            else:

                # Use symmetry to not double calculate
                beta[iM1, iM2, :] = beta[iM2, iM1, :]
         
    return beta 


def get_beta_NL_1D(emu, Mh, mass, ks, z, force_to_zero=0, mass_variable='Mass', knl=5.):
    '''
    One-dimensional Beta_NL function, function: B^NL(Mh, M, k)
    TODO: Change two-dimensional version to accept two separate mass arguments and get rid of this version
    '''
    # Parameters
    klin = np.array([klin_BNL]) # klin must be a numpy array
    Mmin = minimum_halo_mass(emu)

    # Set array name sensibly
    if mass_variable == 'Mass':
        Ms = mass
    elif mass_variable == 'Radius':
        Rs = mass
        Ms = mass_R(emu, Rs)
    elif mass_variable == 'nu':
        nus = mass
        Ms = mass_nu(emu, nus, z)
    else:
        raise ValueError('Error, mass variable for beta_NL not recognised')

    # Linear power
    Pk_lin = emu.get_pklin_from_z(ks, z)
    Pk_klin = emu.get_pklin_from_z(klin, z)
    index_klin, _ = util.findClosestIndex(klin, ks)
    bh = get_linear_halo_bias(emu, Mh, z, klin, Pk_klin)
    
    # Calculate beta_NL by looping over mass arrays
    beta = np.zeros((len(Ms), len(ks)))  
    for iM, M in enumerate(Ms):

        if M < Mmin:
            beta[iM, :] = 0.
        else:
            
            # Calculate beta_NL
            b = get_linear_halo_bias(emu, M, z, klin, Pk_klin)
            Pk_hh = emu.get_phh_mass(ks, Mh, M, z)
            beta[iM, :] = -1.+Pk_hh/(bh*b*Pk_lin)

            # Force Beta_NL to be zero at large scales if necessary
            if force_to_zero != 0:
                Pk_hh0 = emu.get_phh_mass(klin, Mh, M, z)
                db = Pk_hh0/(bh*b*Pk_klin)-1.
                if force_to_zero == 1:
                    beta[iM, :] = beta[iM, :]-db  # Additive correction
                elif force_to_zero == 2:
                    beta[iM, :] = (beta[iM, :]+1.)/(db+1.)-1. # Multiplicative correction
                elif force_to_zero == 3:
                    beta[iM, :index_klin] = beta[iM, index_klin] # for k<klin use the value for klin or the closest to that
                elif force_to_zero == 4:
                    beta[iM, :index_klin] = 0.0 # for k<klin set all values to zero
                    # beta[iM1, iM2, index_knl:] = 0.0 
                elif force_to_zero == 5:
                    beta[iM, :] = beta[iM, :]*(1.-np.exp(-(ks/klin)**2.)) #Smoothly go to zero at k~klin
                else:
                    raise ValueError('force_to_zero not set correctly, choose from 0-5')

    return beta 


# def beta_match_metric(beta1, beta2):
#     '''
#     Single number to quantify how well two different betas match across k, M
#     Clearly the result will depend on the spacing of the M and k values
#     TODO: Probably should compute an RMS integral, but hopefully sum is sufficient
#     '''
#     diff = (beta1-beta2)**2
#     sigma = np.sqrt(diff.sum()/diff.size)
#     return sigma


# def beta_match_metric_k(beta1, beta2):
#     '''
#     Single number to quantify how well two different betas match across M for each k
#     Clearly result will depend on the spacing of the M values
#     TODO: Probably should compute an RMS integral, but hopefully sum is sufficient
#     '''
#     diff = (beta1-beta2)**2
#     nk = len(diff[1, 1, :])
#     sigma = np.zeros(nk)
#     for ik in range(nk):
#         sigma[ik] = np.sqrt(diff[:, :, ik].sum()/diff[:, :, ik].size)
#     return sigma

### ###

### Rescaling ###

# def calculate_rescaling_params(emu_ori, emu_tgt, z_tgt, M1_tgt, M2_tgt):
#     '''
#     Calculates the AW10 rescaling parameters to go from the original cosmology to the target cosmology
#     '''
#     R1_tgt = Radius_M(emu_tgt, M1_tgt)
#     R2_tgt = Radius_M(emu_tgt, M2_tgt)

#     s, sm, z = utility.calculate_AW10_rescaling_parameters(z_tgt, R1_tgt, R2_tgt, 
#                                                          lambda Ri, zi: sigma_R(emu_ori, Ri, zi),
#                                                          lambda Ri, zi: sigma_R(emu_tgt, Ri, zi),
#                                                          emu_ori.cosmo.get_Omega0(),
#                                                          emu_tgt.cosmo.get_Omega0(),
#                                                         )
#     return (s, sm, z)

### HOD ###

def get_Pk_gg(emu, k, redshift):
    '''
    Compute galaxy power spectrum :math:`\\P_\mathrm{gg}(k)`.
    Args:
        k (numpy array): 3 dimensional separation in :math:`h^{-1}\mathrm{Mpc}`
        redshift (float): redshift at which the galaxies are located
    Returns:
        numpy array: galaxy power spectrum
    '''
    from scipy.interpolate import InterpolatedUnivariateSpline as ius
    emu._check_update_redshift(redshift)
    emu._compute_p_1hcs(redshift)
    emu._compute_p_1hss(redshift)
    emu._compute_p_2hcc(redshift)
    emu._compute_p_2hcs(redshift)
    emu._compute_p_2hss(redshift)
    p_tot_1h = 2.*emu.p_1hcs+emu.p_1hss
    p_tot_2h = emu.p_2hcc+2.*emu.p_2hcs+emu.p_2hss
    p_gg = ius(emu.fftlog_1h.k, p_tot_1h)(k)+ius(emu.fftlog_2h.k, p_tot_2h)(k)
    return p_gg


def get_Pk_gm(emu, k, redshift):
    '''
    Compute galaxy matter power spectrum P_gm.
    Args:
        k (numpy array): 2 dimensional projected separation in :math:`h^{-1}\mathrm{Mpc}`
        redshift (float): redshift at which the lens galaxies are located
    Returns:
        numpy array: excess surface density in :math:`h M_\odot \mathrm{pc}^{-2}`
    '''
    from scipy.interpolate import InterpolatedUnivariateSpline as ius
    emu._check_update_redshift(redshift)
    emu._compute_p_cen(redshift)
    emu._compute_p_cen_off(redshift)
    emu._compute_p_sat(redshift)
    p_tot = emu.p_cen+emu.p_cen_off+emu.p_sat
    p_gm = ius(emu.fftlog_1h.k, p_tot)(k)
    return p_gm

### ###