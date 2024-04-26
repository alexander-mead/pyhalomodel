# Standard imports
import numpy as np
from scipy.integrate import quad

# Project imports
from . import constants as const
from . import utility as util

# Constants
Dv0 = 18.*np.pi**2  # Delta_v = ~178, EdS halo virial overdensity
# delta_c = ~1.686' EdS linear collapse threshold
dc0 = (3./20.)*(12.*np.pi)**(2./3.)

# Parameters
xmin_Tk = 1e-5  # Scale at which to switch to Taylor expansion approximation in tophat Fourier functions

### Backgroud ###


def redshift_from_scalefactor(a):
    return -1.+1./a


def scalefactor_from_redshift(z):
    return 1./(1.+z)


def comoving_matter_density(Om_m: float) -> float:
    '''
    Comoving matter density, not a function of time [Msun/h / (Mpc/h)^3]
    args:
        Om_m: Cosmological matter density (at z=0)
    '''
    return const.rho_critical*Om_m

### ###

### Linear perturbations ###


def _Tophat_k(x: np.ndarray) -> np.ndarray:
    '''
    Fourier transform of a tophat function.
    args:
        x: Usually kR
    '''
    xmin = xmin_Tk
    return np.where(np.abs(x) < xmin, 1.-x**2/10., (3./x**3)*(np.sin(x)-x*np.cos(x)))


def _dTophat_k(x: np.ndarray) -> np.ndarray:
    '''
    Derivative of the tophat Fourier transform function
    args:
        x: Usually kR
    '''
    xmin = xmin_Tk
    return np.where(np.abs(x) < xmin, -x/5.+x**3/70., (3./x**4)*((x**2-3.)*np.sin(x)+3.*x*np.cos(x)))


def sigmaR(Rs: np.ndarray, Pk_lin: callable, kmin=1e-5, kmax=1e5, nk=int(1e5), integration_type='brute') -> np.ndarray:
    '''
    Get the square-root of the variance, sigma(R), in the density field
    at comoving Lagrangian scale R
    '''
    if integration_type == 'brute':
        sigmaR_arr = _sigmaR_brute_log(Rs, Pk_lin, kmin=kmin, kmax=kmax, nk=nk)
    elif integration_type == 'quad':
        sigmaR_arr = _sigmaR_quad(Rs, Pk_lin)
    else:
        print('Not a recognised integration_type. Try one of the following:')
        print('brute for brute force integration')
        print('quad for the general purpose quad integration')
    return sigmaR_arr


def _sigmaR_integrand(k: np.array, R: float, Pk: callable) -> np.ndarray:
    '''
    Integrand for calculating sigma(R)
    Note that k can be a float or an arraay here
    args:
        k: Fourier wavenumber (or array of these) [h/Mpc]
        R: Comoving Lagrangian radius [Mpc/h]
        Pk: Function of k to evaluate the linear power spectrum
    '''
    return Pk(k)*(k**2)*_Tophat_k(k*R)**2


def _sigmaR_brute_log(R: float, Pk: callable, kmin=1e-5, kmax=1e5, nk=int(1e5)) -> float:
    '''
    Brute force integration, this is only slightly faster than using a loop
    args:
        R: Comoving Lagrangian radius [Mpc/h]
        Pk: Function of k to evaluate the linear power spectrum
        kmin: Minimum wavenumber [h/Mpc]
        kmax: Maximum wavenumber [h/Mpc]
        nk: Number of bins in wavenumber
    '''
    k = util.logspace(kmin, kmax, nk)
    dlnk = np.log(k[1]/k[0])

    def sigmaR_vec(R, Pk):
        sigmaR = np.sqrt(sum(dlnk*k*_sigmaR_integrand(k, R, Pk))/(2.*np.pi**2))
        return sigmaR
    sigma_func = np.vectorize(sigmaR_vec, excluded=['Pk'])
    return sigma_func(R, Pk)


def _sigmaR_quad(R: float, Pk: callable) -> float:
    '''
    Quad integration
    args:
        R: Comoving Lagrangian radius [Mpc/h]
        Pk: Function of k to evaluate the linear power spectrum
    '''
    def sigmaR_vec(R: float, Pk: callable):
        kmin, kmax = 0., np.inf
        sigma_squared, _ = quad(lambda k: _sigmaR_integrand(k, R, Pk),
                                kmin, kmax, epsrel=1e-4, epsabs=0.)
        sigma = np.sqrt(sigma_squared/(2.*np.pi**2))
        return sigma
    sigma_func = np.vectorize(sigmaR_vec, excluded=['Pk'])
    return sigma_func(R, Pk)


def _dsigmaR_integrand(k: float, R: float, Pk) -> float:
    return Pk(k)*(k**3)*_Tophat_k(k*R)*_dTophat_k(k*R)


def dlnsigma2_dlnR(R: float, Pk: callable) -> float:
    '''
    Calculates d(ln sigma^2)/d(ln R) by integration
    '''
    def dsigmaR_vec(R, Pk):
        kmin, kmax = 0., np.inf  # Evaluate the integral and convert to a nicer form
        dsigma, _ = quad(lambda k: _dsigmaR_integrand(k, R, Pk),
                         kmin, kmax, epsrel=1e-4, epsabs=0.)
        dsigma = R*dsigma/(np.pi*_sigmaR_quad(R, Pk))**2
        return dsigma
    dsigma_func = np.vectorize(dsigmaR_vec, excluded=['Pk'])
    return dsigma_func(R, Pk)

### ###

### Haloes ###


def Lagrangian_radius(M: float, Om_m: float) -> float:
    '''
    Radius [Mpc/h] of a sphere containing mass M in a homogeneous universe
    args:
        M: Halo mass [Msun/h]
        Om_m: Cosmological matter density (at z=0)
    '''
    return np.cbrt(3.*M/(4.*np.pi*comoving_matter_density(Om_m)))


def mass(R: float, Om_m: float) -> float:
    '''
    Mass [Msun/h] contained within a sphere of radius 'R' [Mpc/h] in a homogeneous universe
    '''
    return (4./3.)*np.pi*R**3*comoving_matter_density(Om_m)

### ###

### Spherical collapse ###


def dc_NakamuraSuto(Om_mz: float) -> float:
    '''
    LCDM fitting function for the critical linear collapse density from Nakamura & Suto
    (1997; https://arxiv.org/abs/astro-ph/9612074)
    Cosmology dependence is very weak
    '''
    return dc0*(1.+0.012299*np.log10(Om_mz))


def Dv_BryanNorman(Om_mz: float) -> float:
    '''
    LCDM fitting function for virial overdensity from Bryan & Norman
    (1998; https://arxiv.org/abs/astro-ph/9710107)
    Note that here Dv is defined relative to background matter density,
    whereas in paper it is relative to critical density
    For Omega_m = 0.3 LCDM Dv ~ 330.
    '''
    x = Om_mz-1.
    Dv = Dv0+82.*x-39.*x**2
    return Dv/Om_mz

### ###
