# Standard imports
import numpy as np
import warnings

# Other imports
from dark_emulator import darkemu

# My imports
from . import constants as const
from . import utility as util
from . import cosmology as cosm

# Constants
dc = 1.686      # Collapse threshold for nu definition
Dv = 200.       # Spherical-overdensity halo definition from Dark Emualtor (fixed)
np_min = 200    # Minimum number of halo particles
npart = 2048    # Cube root of number of simulation particles
Lbox_HR = 1000. # Box size for high-resolution simulations [Mpc/h]
Lbox_LR = 2000. # Box size for low-resolution simulations [Mpc/h]

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

# Beta-NL
# Source of linear halo bias
# bias_def = 'bias': From emulator 'bias' function
bias_def = 'halo-halo' # From emulator halo-halo spectrum at large scale
# ibias_def = 'halo-matter' # From emulator halo-matter spectrum at large scale
klin_BNL = 0.02 # Large 'linear' scale [h/Mpc]

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
        self.m_nu = self.wnu*const.neutrino_constant
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

        print('Dark Quest fixed parameters')
        print('omega_nu: %1.4f' % (self.wnu))
        print('Omega_k: %1.4f' % (self.Om_k))
        print()

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


def init_emulator(cpar):
    '''
    Initialise the emulator for a given set of cosmological parameters
    cpar: My verion of Dark Quest cosmology object
    '''
    # Start Dark Quest
    print('Initialize Dark Quest')
    emu = darkemu.base_class()
    print()

    # Initialise emulator
    cpar.print()
    cparam = cpar.cosmology_array()
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
    Mbox_HR = _comoving_matter_density(emu)*Lbox_HR**3
    mmin = Mbox_HR*np_min/npart**3
    return mmin


def _comoving_matter_density(emu):
    '''
    Comoving matter density [(Msun/h)/(Mpc/h)^3]
    '''
    Om_m = emu.cosmo.get_Omega0()
    rhom = cosm.comoving_matter_density(Om_m)
    return rhom


def _mass_R(emu, R):
    '''
    Mass enclosed within comoving radius R [Msun/h]
    '''
    Om_m = emu.cosmo.get_Omega0()
    return cosmology.mass(R, Om_m)


def _mass_nu(emu, nu, z):

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


def _get_bias_mass(emu, M, redshift):
    '''
    Linear halo bias: b(M, z)
    Taken from the pre-release version of Dark Quest given to me by Takahiro
    I am not sure why this functionality was omitted from the final version
    '''
    Mp = M*1.01
    Mm = M*0.99
    logdensp = np.log10(emu.mass_to_dens(Mp, redshift))
    logdensm = np.log10(emu.mass_to_dens(Mm, redshift))
    bp = emu.get_bias(logdensp, redshift)
    bm = emu.get_bias(logdensm, redshift)
    return (bm*10**logdensm - bp*10**logdensp) / (10**logdensm-10**logdensp)


def get_linear_halo_bias(emu, M, z, klin, Pk_klin, method=bias_def):
    '''
    Linear halo bias
    '''
    if method == 'bias':
        b = _get_bias_mass(emu, M, z)[0]
    elif method == 'halo-halo':
        b = np.sqrt(emu.get_phh_mass(klin, M, M, z)/Pk_klin)
    elif method == 'halo-matter':
        b = emu.get_phm_mass(klin, M, z)/Pk_klin
    else:
        raise ValueError('Linear bias recipe not recognised')
    return b


def get_halo_cross_spectrum_coefficient(emu, ks, M1, M2, z):
    '''
    Cross correlation coefficient between halo masses
    '''
    P12 = emu.get_phh_mass(ks, M1, M2, z)
    P11 = emu.get_phh_mass(ks, M1, M1, z)
    P22 = emu.get_phh_mass(ks, M2, M2, z)
    if (P11 < 0.).any() or (P22 < 0.).any():
        warnings.warn('Negative values in halo auto power, M1=M2='+'%0.2f'%(np.log10(M1)), RuntimeWarning)
    return P12/np.sqrt(P11*P22)

### ###

### Non-linear halo bias ###

def get_beta_NL(emu, mass, ks, z, force_to_zero=0, mass_variable='mass', knl=5.):
    '''
    Beta_NL function, function: B^NL(M1, M2, k)
    TODO: Change to accept two separate mass arguments and merge with beta_NL_1D?
    TODO: knl unused
    TODO: Use strings for different force_to_zero methods
    '''
    # Parameters
    klin = np.array([klin_BNL]) # klin must be a numpy array
    
    # Set array name sensibly
    if mass_variable == 'mass':
        Ms = mass
    elif mass_variable == 'radius':
        Rs = mass
        Ms = _mass_R(emu, Rs)
    elif mass_variable == 'nu':
        nus = mass
        Ms = _mass_nu(emu, nus, z)
    else:
        raise ValueError('Error, mass variable for beta_NL not recognised')
    
    # Linear power
    Pk_lin = emu.get_pklin_from_z(ks, z)
    Pk_klin = emu.get_pklin_from_z(klin, z)
    index_klin, _ = util.find_closest_index_value(klin, ks)
    
    # Calculate beta_NL by looping over mass arrays
    beta = np.zeros((len(ks), len(Ms), len(Ms)))
    for iM1, M1 in enumerate(Ms):

        # Linear halo bias
        b1 = get_linear_halo_bias(emu, M1, z, klin, Pk_klin)

        for iM2, M2 in enumerate(Ms):

            if iM2 >= iM1:

                # Create beta_NL
                b2 = get_linear_halo_bias(emu, M2, z, klin, Pk_klin)
                Pk_hh = emu.get_phh_mass(ks, M1, M2, z)
                beta[:, iM1, iM2] = Pk_hh/(b1*b2*Pk_lin)-1.

                # Force Beta_NL to be zero at large scales if necessary
                if force_to_zero != 0:
                    Pk_hh0 = emu.get_phh_mass(klin, M1, M2, z)
                    db = Pk_hh0/(b1*b2*Pk_klin)-1.
                    if force_to_zero == 1:
                        beta[:, iM1, iM2] = beta[:, iM1, iM2]-db # Additive correction
                    elif force_to_zero == 2:
                        beta[:, iM1, iM2] = (beta[:, iM1, iM2]+1.)/(db+1.)-1. # Multiplicative correction
                    elif force_to_zero == 3:
                        # print('setting beta_nl=beta_nl(k='+'%0.3f' % klin_closest+') for k<'+ '%0.3f' % klin_closest)
                        beta[:index_klin, iM1, iM2, ] = beta[index_klin, iM1, iM2] # for k<klin use the value for klin or the closest to that
                    elif force_to_zero == 4:
                        # print('setting beta_nl=0 for k<'+ '%0.3f' % klin_closest)
                        beta[:index_klin, iM1, iM2, ] = 0.0 # for k<klin set all values to zero
                        # beta[iM1, iM2, index_knl:] = 0.0 
                    elif force_to_zero == 5:
                        beta[:, iM1, iM2] = beta[:, iM1, iM2]*(1.-np.exp(-(ks/klin)**2)) #Smoothly go to zero at k~klin
                    elif force_to_zero == 6:
                        # like 5 but with dependence on M1 and M2
                        if (iM1 == iM2) or (np.log10(M2/M1)<1):
                            truncate_bnl= (1.-np.exp(-(ks/klin)))
                        else:
                            truncate_bnl= (1.-np.exp(-(ks/klin)*np.log10(M2/M1)))
                        beta[:, iM1, iM2] = beta[:, iM1, iM2]*truncate_bnl #Change to smoothly going to zero
                    else:
                        raise ValueError('force_to_zero not set correctly, choose from 0-6')

            else:

                # Use symmetry to not double calculate
                beta[:, iM1, iM2] = beta[:, iM2, iM1]
         
    return beta 


def get_beta_NL_1D(emu, Mh, mass, ks, z, force_to_zero=0, mass_variable='mass', knl=5.):
    '''
    One-dimensional Beta_NL function, function: B^NL(Mh, M, k)
    TODO: Change two-dimensional version to accept two separate mass arguments and get rid of this version
    '''
    # Parameters
    klin = np.array([klin_BNL]) # klin must be a numpy array
    Mmin = minimum_halo_mass(emu)

    # Set array name sensibly
    if mass_variable == 'mass':
        Ms = mass
    elif mass_variable == 'radius':
        Rs = mass
        Ms = _mass_R(emu, Rs)
    elif mass_variable == 'nu':
        nus = mass
        Ms = _mass_nu(emu, nus, z)
    else:
        raise ValueError('Error, mass variable for beta_NL not recognised')

    # Linear power
    Pk_lin = emu.get_pklin_from_z(ks, z)
    Pk_klin = emu.get_pklin_from_z(klin, z)
    index_klin, _ = util.find_closest_index_value(klin, ks)
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

### ###

### HOD ###

def get_Pk_gg(emu, k, redshift):
    '''
    Compute galaxy power spectrum P_gg
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
    Compute galaxy matter power spectrum P_gm
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