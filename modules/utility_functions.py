# Utility functions for the Halo Model calculations

import numpy as np
import camb
import constants as const
import scipy.integrate as integrate

# Fourier transform of a tophat function.
# Parameters
xmin_Tk = 1e-5
def Tophat_k(x):
    # Normalised tophat Fourier transform function such that T(x=0)=1
    xmin = xmin_Tk
    return np.where(np.abs(x)<xmin, 1.-x**2/10., (3./x**3)*(np.sin(x)-x*np.cos(x)))

def dTophat_k(x):
    '''
    Derivative of the tophat Fourier transform function
    '''
    xmin = xmin_Tk
    return np.where(np.abs(x)<xmin, -x/5.+x**3/70., (3./x**4)*((x**2-3.)*np.sin(x)+3.*x*np.cos(x)))


def get_sigmaR(Rs,Pk_lin,kmin=1e-5,kmax=1e5,nk=1e5,integration_type='brute'):
    # 
    if integration_type == 'brute':
        sigmaR_arr = sigmaR_brute_log(Rs,Pk_lin,kmin=kmin,kmax=kmax,nk=nk)
    elif integration_type == 'quad':
        sigmaR_arr = sigma_R_quad(Rs,Pk_lin)
#     elif integration_type == 'gauleg':
#         sigmaR_arr = sigma_R_gauleg(Rs,Pk_lin)
    elif integration_type == 'camb':
        sigmaR_arr = sigma_R_camb(Rs,Pk_lin)
    else:
        print('not a recognised integration_type. Try one of the following:')
        print('brute for brute force integration')
        print('quad for the general purpose quad integration')
        # print('gauleg for Gaussian Legendre integration')
    return sigmaR_arr


def sigma_integrand(k,R=1,Power_k='None'):
    return Power_k(0,k)*(k**2)*Tophat_k(k*R)**2

# Brute force integration, this is only slightly faster than using a loop
def sigmaR_brute_log(R,Power_k,kmin=1e-5,kmax=1e5,nk=1e5):
	k_arr = np.logspace(np.log10(kmin),np.log10(kmax),int(nk))
	dln_k = np.log(k_arr[1]/k_arr[0])
	def sigma_R_vec(R):
		def sigma_integrand(k):
			return Power_k(k)*(k**2)*Tophat_k(k*R)**2

		sigmaR= np.sqrt(sum(dln_k*k_arr*sigma_integrand(k_arr))/(2.*np.pi**2))
		return sigmaR

	# Note that this is a function  
	sigma_func = np.vectorize(sigma_R_vec, excluded=['Power_k']) 

	# This is the function evaluated
	return sigma_func(R)

# Quad integration
def sigma_R_quad(R, Power_k):

    def sigma_R_vec(R):
        def sigma_integrand(k):
            return Power_k(k)*(k**2)*Tophat_k(k*R)**2

        # k range for integration (could be restricted if necessary)
        kmin = 0.
        kmax = np.inf

        # Evaluate the integral and convert to a nicer form
        sigma_squared, _ = integrate.quad(sigma_integrand, kmin, kmax)
        sigma = np.sqrt(sigma_squared/(2.*np.pi**2))
        return sigma

    # Note that this is a function  
    sigma_func = np.vectorize(sigma_R_vec, excluded=['Power_k']) 

    # This is the function evaluated
    return sigma_func(R)

# Get sigma(R) from CAMB: note CAMB uses fortran to do the calculations and is therefore faster
def sigma_R_camb(R, results):
	sigmaRs = results.get_sigmaR(R, hubble_units=True, return_R_z=False)
	return sigmaRs


def dlnsigma2_dlnR(R, Power_k):
    '''
    Calculates d(ln sigma^2)/d(ln R) by integration
    '''
    def dsigma_R_vec(R):
        def dsigma_integrand(k):
            return Power_k(k)*(k**3)*Tophat_k(k*R)*dTophat_k(k*R)

        # Evaluate the integral and convert to a nicer form
        kmin = 0. 
        kmax = np.inf # Integration range

        dsigma, _ = integrate.quad(dsigma_integrand, kmin, kmax)
        dsigma = R*dsigma/(np.pi*sigma_R(R, Power_k))**2
        return dsigma

    # Note that this is a function
    dsigma_func = np.vectorize(dsigma_R_vec, excluded=['Power_k'])

    # This is the function evaluated
    return dsigma_func(R)


def sigma_R(R, Power_k):

    def sigma_R_vec(R):

        def sigma_integrand(k):
            return Power_k(k)*(k**2)*Tophat_k(k*R)**2

        # k range for integration (could be restricted if necessary)
        kmin = 0.
        kmax = np.inf

        # Evaluate the integral and convert to a nicer form
        sigma_squared, _ = integrate.quad(sigma_integrand, kmin, kmax)
        sigma = np.sqrt(sigma_squared/(2.*np.pi**2))
        return sigma

    # Note that this is a function  
    sigma_func = np.vectorize(sigma_R_vec, excluded=['Power_k']) 

    # This is the function evaluated
    return sigma_func(R) 

def Radius_M(M, Om_m):
	'''
	Radius of a sphere containing mass M in a homogeneous universe
	'''
	return np.cbrt(3.*M/(4.*np.pi*comoving_matter_density(Om_m)))

def Mass_R(R, Om_m):
	'''
	Mass contained within a sphere of radius 'R' in a homogeneous universe
	'''
	return Mass_R(R, Om_m)

def comoving_matter_density(Om_m):
	'''
	Comoving matter density, not a function of time [Msun/h / (Mpc/h)^3]
	'''
	return const.rhoc*Om_m


def scale_factor_z(z):
	'''
	Scale factor at redshift z: 1/a = 1+z
	'''
	return 1./(1.+z)


def Hubble_function(Om_r,Om_m,Om_w,Om_v, a):
	'''
	Hubble parameter, $(\dot{a}/a)^2$, normalised to unity at a=1
	'''
	H2=(Om_r*a**-4)+(Om_m*a**-3)+Om_w*X_de(a)+Om_v+(1.-Om)*a**-2
	return np.sqrt(H2)


def X_de(ide, a, w=None, wa=None,nw=None):
	'''
	Dark energy energy density, normalised to unity at a=1
	ide:
	0 - Fixed w = -1
	1 - w(a)CDM
	2 - wCDM
	5 - IDE II
	'''
	if(ide == 0):
		# Fixed w = -1, return ones
		return np.full_like(a, 1.) # Make numpy compatible
	if(ide == 1):
	# w(a)CDM
		return a**(-3.*(1.+w+wa))*np.exp(-3.*wa*(1.-a))
	elif(ide == 2):
		# wCDM same as w(a)CDM if w(a)=0
		return a**(-3.*(1.+w))
	# elif(ide == 5):
	#     f1=(a/a1)**nw+1.
	#     f2=(1./a1)**nw+1.
	#     f3=(1./a2)**nw+1.
	#     f4=(a/a2)**nw+1.
	#     return ((f1/f2)*(f3/f4))**(-6./nw)
	else:
		raise ValueError('ide not recognised')


def calculate_AW10_rescaling_parameters(z_tgt, R1_tgt, R2_tgt, sigma_Rz_ogn, sigma_Rz_tgt, Om_m_ogn, Om_m_tgt):

    from scipy.optimize import fmin

    def rescaling_cost_function(s, z, z_tgt, R1_tgt, R2_tgt, sigma_Rz_ogn, sigma_Rz_tgt):

        # Severely punish negative z
        if (z < 0.):
            return AW10_future_punishment

        def integrand(R):
            return (1./R)*(1.-sigma_Rz_ogn(R/s, z)/sigma_Rz_tgt(R, z_tgt))**2

        integral, _ = integrate.quad(integrand, R1_tgt, R2_tgt)
        cost = integral/np.log(R2_tgt/R1_tgt)
        return cost

    s0 = 1.
    z0 = z_tgt

    s, z = fmin(lambda x: rescaling_cost_function(x[0], x[1], z_tgt, R1_tgt, R2_tgt, sigma_Rz_ogn, sigma_Rz_tgt), [s0, z0])
    sm = (Om_m_tgt/Om_m_ogn)*s**3

    # Warning
    if z < 0.:
        print('Warning: Rescaling redshift is in the future for the original cosmology')

    return s, sm, z

def log_derivative(f, x, dx=1e-3):
	'''
	Calculate the logarithmic derivative of f(x) at the point x: dln(f)/dln(x)
	'''
	from numpy import log
	dlnf = log(f(x+dx/2.)/f(x-dx/2.)) # Two-sided difference in numerator
	dlnx = log((x+dx/2.)/(x-dx/2.)) # Two-sided (is this necessary?)
	#dlnx = log(1.+dx/x) # Using this is probably fine; for dx<<x they are equal
	return dlnf/dlnx

def logspace(xmin, xmax, nx):
    '''
    Return a logarithmically spaced range of numbers
    Numpy version is specifically base10, which is insane since log spacing is independent of base
    '''
    from numpy import logspace, log10
    return logspace(log10(xmin), log10(xmax), nx)

