# Standard imports
import numpy as np
import scipy.integrate as integrate

# Project imports
import constants as const

# To-do list
# TODO: Remove commented-out stuff

# Constants
# TODO: Remove if not keeping relations
# Dv0 = 18.*np.pi**2                  # Delta_v = ~178, EdS halo virial overdensity
# dc0 = (3./20.)*(12.*np.pi)**(2./3.) # delta_c = ~1.686' EdS linear collapse threshold

# Parameters
xmin_Tk = 1e-5 # Scale at which to switch to Taylor expansion approximation

def _Tophat_k(x:np.ndarray) -> np.ndarray:
    '''
    Fourier transform of a tophat function.
    args:
        x: Usually kR
    '''
    xmin = xmin_Tk
    return np.where(np.abs(x)<xmin, 1.-x**2/10., (3./x**3)*(np.sin(x)-x*np.cos(x)))


def _dTophat_k(x:np.ndarray) -> np.ndarray:
    '''
    Derivative of the tophat Fourier transform function
    args:
        x: Usually kR
    '''
    xmin = xmin_Tk
    return np.where(np.abs(x)<xmin, -x/5.+x**3/70., (3./x**4)*((x**2-3.)*np.sin(x)+3.*x*np.cos(x)))


def sigmaR(Rs:np.ndarray, Pk_lin, kmin=1e-5, kmax=1e5, nk=1e5, integration_type='brute') -> np.ndarray:
    '''
    Get the square-root of the variance, sigma(R), in the density field
    at comoving Lagrangian scale R
    ''' 
    if integration_type == 'brute':
        sigmaR_arr = _sigmaR_brute_log(Rs, Pk_lin, kmin=kmin, kmax=kmax, nk=nk)
    elif integration_type == 'quad':
        sigmaR_arr = _sigmaR_quad(Rs, Pk_lin)
    #elif integration_type == 'gauleg':
    #    sigmaR_arr = sigma_R_gauleg(Rs,Pk_lin)
    elif integration_type == 'camb':
        sigmaR_arr = _sigmaR_camb(Rs, Pk_lin)
    else:
        print('Not a recognised integration_type. Try one of the following:')
        print('brute for brute force integration')
        print('quad for the general purpose quad integration')
        # print('gauleg for Gaussian Legendre integration')
    return sigmaR_arr


def _sigmaR_integrand(k, R:float, Pk) -> np.ndarray:
    '''
    Integrand for calculating sigma(R)
    Note that k can be a float or an arraay here
    args:
        k: Fourier wavenumber (or array of these) [h/Mpc]
        R: Comoving Lagrangian radius [Mpc/h]
        Pk: Function of k to evaluate the linear power spectrum
    '''
    return Pk(k)*(k**2)*_Tophat_k(k*R)**2


def _sigmaR_brute_log(R:float, Pk, kmin=1e-5, kmax=1e5, nk=1e5) -> float:
    '''
    Brute force integration, this is only slightly faster than using a loop
    args:
        R: Comoving Lagrangian radius [Mpc/h]
        Pk: Function of k to evaluate the linear power spectrum
        kmin: Minimum wavenumber [h/Mpc]
        kmin: Maximum wavenumber [h/Mpc]
        nk: Number of bins in wavenumber
    '''
    k_arr = np.logspace(np.log10(kmin), np.log10(kmax), int(nk))
    dln_k = np.log(k_arr[1]/k_arr[0])
    def sigmaR_vec(R, Pk):
        sigmaR= np.sqrt(sum(dln_k*k_arr*_sigmaR_integrand(k_arr, R, Pk))/(2.*np.pi**2))
        return sigmaR
    sigma_func = np.vectorize(sigmaR_vec, excluded=['Pk']) # Note that this is a function
    return sigma_func(R, Pk)
 

def _sigmaR_quad(R:float, Pk) -> float:
    '''
    Quad integration
    args:
        R: Comoving Lagrangian radius [Mpc/h]
        Pk: Function of k to evaluate the linear power spectrum
    '''
    def sigmaR_vec(R, Pk):
        kmin, kmax = 0., np.inf
        sigma_squared, _ = integrate.quad(lambda k: _sigmaR_integrand(k, R, Pk), kmin, kmax)
        sigma = np.sqrt(sigma_squared/(2.*np.pi**2))
        return sigma
    sigma_func = np.vectorize(sigmaR_vec, excluded=['Pk']) # Note that this is a function  
    return sigma_func(R, Pk)


def _sigmaR_camb(R:float, results) -> np.ndarray:
    '''
    Get sigma(R) from CAMB: note CAMB uses Fortran to do the calculations and is therefore faster
    '''
    sigmaRs = results.get_sigmaR(R, hubble_units=True, return_R_z=False)
    return sigmaRs


def _dsigmaR_integrand(k:float, R:float, Pk) -> float:
    return Pk(k)*(k**3)*_Tophat_k(k*R)*_dTophat_k(k*R)


def dlnsigma2_dlnR(R:float, Pk) -> float:
    '''
    Calculates d(ln sigma^2)/d(ln R) by integration
    '''
    def dsigmaR_vec(R, Pk):
        kmin, kmax = 0., np.inf # Evaluate the integral and convert to a nicer form
        dsigma, _ = integrate.quad(lambda k: _dsigmaR_integrand(k, R, Pk), kmin, kmax)
        dsigma = R*dsigma/(np.pi*_sigmaR_quad(R, Pk))**2
        return dsigma
    dsigma_func = np.vectorize(dsigmaR_vec, excluded=['Pk']) # Note that this is a function
    return dsigma_func(R, Pk)


def comoving_matter_density(Om_m:float) -> float:
    '''
    Comoving matter density, not a function of time [Msun/h / (Mpc/h)^3]
    args:
        Om_m: Cosmological matter density (at z=0)
    '''
    return const.rhoc*Om_m


def Lagrangian_radius(M:float, Om_m:float) -> float:
    '''
    Radius [Mpc/h] of a sphere containing mass M in a homogeneous universe
    args:
        M: Halo mass [Msun/h]
        Om_m: Cosmological matter density (at z=0)
    '''
    return np.cbrt(3.*M/(4.*np.pi*comoving_matter_density(Om_m)))


def mass(R:float, Om_m:float) -> float:
    '''
    Mass [Msun/h] contained within a sphere of radius 'R' [Mpc/h] in a homogeneous universe
    '''
    return (4./3.)*np.pi*R**3*comoving_matter_density(Om_m)


# def scale_factor_z(z):
#     '''
#     Scale factor at redshift z: 1/a = 1+z
#     '''
#     return 1./(1.+z)


# def Hubble_function(Om_r, Om_m, Om_w, Om_v, a):
# 	'''
# 	Hubble parameter, $(\dot{a}/a)^2$, normalised to unity at a=1
# 	'''
# 	H2=(Om_r*a**-4)+(Om_m*a**-3)+Om_w*X_de(a)+Om_v+(1.-Om)*a**-2
# 	return np.sqrt(H2)


# def X_de(ide, a, w=None, wa=None,nw=None):
# 	'''
# 	Dark energy energy density, normalised to unity at a=1
# 	ide:
# 	0 - Fixed w = -1
# 	1 - w(a)CDM
# 	2 - wCDM
# 	5 - IDE II
# 	'''
# 	if(ide == 0):
# 		# Fixed w = -1, return ones
# 		return np.full_like(a, 1.) # Make numpy compatible
# 	if(ide == 1):
# 	# w(a)CDM
# 		return a**(-3.*(1.+w+wa))*np.exp(-3.*wa*(1.-a))
# 	elif(ide == 2):
# 		# wCDM same as w(a)CDM if w(a)=0
# 		return a**(-3.*(1.+w))
# 	# elif(ide == 5):
# 	#     f1=(a/a1)**nw+1.
# 	#     f2=(1./a1)**nw+1.
# 	#     f3=(1./a2)**nw+1.
# 	#     f4=(a/a2)**nw+1.
# 	#     return ((f1/f2)*(f3/f4))**(-6./nw)
# 	else:
# 		raise ValueError('ide not recognised')


# def calculate_AW10_rescaling_parameters(z_tgt, R1_tgt, R2_tgt, sigma_Rz_ogn, sigma_Rz_tgt, Om_m_ogn, Om_m_tgt):

#     from scipy.optimize import fmin

#     def rescaling_cost_function(s, z, z_tgt, R1_tgt, R2_tgt, sigma_Rz_ogn, sigma_Rz_tgt):

#         # Severely punish negative z
#         if (z < 0.):
#             return AW10_future_punishment

#         def integrand(R):
#             return (1./R)*(1.-sigma_Rz_ogn(R/s, z)/sigma_Rz_tgt(R, z_tgt))**2

#         integral, _ = integrate.quad(integrand, R1_tgt, R2_tgt)
#         cost = integral/np.log(R2_tgt/R1_tgt)
#         return cost

#     s0 = 1.
#     z0 = z_tgt

#     s, z = fmin(lambda x: rescaling_cost_function(x[0], x[1], z_tgt, R1_tgt, R2_tgt, sigma_Rz_ogn, sigma_Rz_tgt), [s0, z0])
#     sm = (Om_m_tgt/Om_m_ogn)*s**3

#     # Warning
#     if z < 0.:
#         print('Warning: Rescaling redshift is in the future for the original cosmology')

#     return s, sm, z


# def dc_NakamuraSuto(Om_mz):
#     '''
#     LCDM fitting function for the critical linear collapse density from Nakamura & Suto 
#     (1997; https://arxiv.org/abs/astro-ph/9612074)
#     Cosmology dependence is very weak
#     '''
#     return dc0*(1.+0.012299*np.log10(Om_mz))


# def Dv_BryanNorman(Om_mz):
#     '''
#     LCDM fitting function for virial overdensity from Bryan & Norman 
#     (1998; https://arxiv.org/abs/astro-ph/9710107)
#     Note that here Dv is defined relative to background matter density, 
#     whereas in paper it is relative to critical density
#     For Omega_m = 0.3 LCDM Dv ~ 330.
#     '''
#     x = Om_mz-1.
#     Dv = Dv0+82.*x-39.*x**2
#     return Dv/Om_mz