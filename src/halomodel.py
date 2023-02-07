# Standard imports
import numpy as np
import scipy.integrate as integrate
import warnings

# Project imports
import utility as util
import cosmology

# To-do list
# TODO: Dedicated test suite against HMcode/HMx for different cosmologies/halo models/redshifts
# TODO: Add more checks for Dv, dc value compatability with mass functions
# TODO: Add covariance between profile amplitudes (low priority; hard and annoying)
# TODO: Change beta(M1, M2, k) -> beta(k, M1, M2)?

# Parameters
dc_rel_tol = 1e-3    # Relative tolerance for checking 'closeness' of delta_c
Dv_rel_tol = 1e-3    # Relative tolerance for checking 'closeness' of Delta_v
Tinker_z_dep = False # Use redshift-dependent parameters for Tinker formula (only calibrated for M200)
Tinker_PBS = False   # Get the bias from the peak-background split, rather than using the calibrated formula

# Halo-model integration scheme for integration in nu
# NOTE: Cannot use Romberg because integration is done in nu, not M, and this will generally be uneven
# NOTE: In HMx I use trapezoid because of fast osciallations, maybe should also use that here?
halo_integration = integrate.trapezoid
#halo_integration = integrate.simps

# W(k) integration scheme integration in r
# TODO: Add FFTlog?
#win_integration = integrate.trapezoid
#win_integration = integrate.simps
#win_integration = integrate.romb # Needs 2^m+1 (integer m) evenly-spaced samples in R
win_integration = integrate.quad
#win_integration = integrate.quadrature
#win_integration = integrate.romberg
nr_win_integration = 1+2**8 # Number of points in r (sample methods)
eps_win_integration = 1e-3 # Integration accuracy (continuous methods)

# Mass function
eps_deriv_mf = 1e-3 # R -> dR for numerical sigma derivative

# beta_NL
do_I11 = True     # Compute and add low M1 and M2 portion of the integral
do_I12_I21 = True # Compute and add low M1 or low M2 portion of the integral

### Class definition ###

class model():
    '''
    Class for halo model
    '''
    def __init__(self, z:float, Om_m:float, name='Tinker et al. (2010)', Dv=200., dc=1.686, verbose=False):
        '''
        Class initialisation; mainly configures mass function and halo bias.\n
        Args:
            z: Redshift
            Om_m: Cosmological matter density
            name: Name of halo model, one of:
                'Press & Schecter (1974)'
                'Sheth & Tormen (1999)'
                'Sheth, Mo & Tormen (2001)'
                'Tinker et al. (2010)'
                'Despali et al. (2016)'
            Dv: Halo overdensity definition
            dc: Halo linear collapse threshold
            verbose: Verbosity
        '''
        # Store internal variables
        self.z = z
        self.a = 1./(1.+z)
        self.Om_m = Om_m
        self.name = name
        self.dc = dc
        self.Dv = Dv
        self.rhom = cosmology.comoving_matter_density(Om_m) # TODO: Change name?

        # Write to screen
        if verbose:
            print('Initialising halo model')
            print('Halo model:', self.name)
            print('Redshift: %1.3f'%(self.z))
            print('Scale factor: %1.3f'%(self.a))
            print('Omega_m(z=0): %1.3f'%(self.Om_m))
            print('delta_c: %1.4f'%(self.dc))
            print('Delta_v: %4.1f'%(self.Dv))

        # Initialise mass functions
        if name == 'Press & Schecter (1974)':
            # No initialisation required for Press & Schecter (1974)
            pass
        elif name in ['Sheth & Tormen (1999)', 'Sheth, Mo & Tormen (2001)']:
            # Sheth & Tormen (1999; https://arxiv.org/abs/astro-ph/9901122)
            # Virial halo definition
            # TODO: Check for inconsistencies with Dv, dc
            from scipy.special import gamma as Gamma
            p = 0.3
            q = 0.707
            self.p_ST = p; self.q_ST = q
            self.A_ST = np.sqrt(2.*q)/(np.sqrt(np.pi)+Gamma(0.5-p)/2**p) # A ~ 0.2161
            if verbose:
                print('p: %1.3f; q: %1.3f; A: %1.4f'%(self.p_ST, self.q_ST, self.A_ST))
            if name in ['Sheth, Mo & Tormen (2001)']:
                self.a_SMT = 0.707
                self.b_SMT = 0.5
                self.c_SMT = 0.6
                if verbose:
                    print('a: %1.3f; b: %1.3f; c: %1.3f'%(self.a_SMT, self.b_SMT, self.c_SMT))
        elif name in ['Despali et al. (2016)']:
            # Despali et al. (2016; https://arxiv.org/abs/1507.05627)
            # Note that this mass function is not correctly normalised
            # Virial halo definition
            # TODO: Check for inconsistencies with Dv, dc
            self.p_ST = 0.2536
            self.q_ST = 0.7689
            self.A_ST = 0.3295*np.sqrt(2.*self.q_ST/np.pi) # Careful with factors of sqrt(2q/pi) in normalisation
            if verbose:
                print('p: %1.3f; q: %1.3f; A: %1.4f'%(self.p_ST, self.q_ST, self.A_ST))
        elif name in ['Tinker et al. (2010)']:
            # Tinker et al. (2010; https://arxiv.org/abs/1001.3162)
            # A range of Delta_v are available
            from math import isclose
            Dv_array = np.array([200., 300., 400., 600., 800., 1200., 1600, 2400., 3200.])
            # Check Delta_v and delta_c values
            if Dv < Dv_array[0] or Dv > Dv_array[-1]:
                warnings.warn('Dv is outside supported range for Tinker et al. (2010)', RuntimeWarning)
            if not isclose(dc, 1.686, rel_tol=dc_rel_tol):
                warnings.warn('dc = 1.686 assumed in Tinker et al. (2010)', RuntimeWarning)
            # Mass function from Table 4
            logDv = np.log(self.Dv)
            logDv_array = np.log(Dv_array)
            alpha_array = np.array([0.368, 0.363, 0.385, 0.389, 0.393, 0.365, 0.379, 0.355, 0.327])
            beta_array = np.array([0.589, 0.585, 0.544, 0.543, 0.564, 0.623, 0.637, 0.673, 0.702])
            gamma_array = np.array([0.864, 0.922, 0.987, 1.09, 1.20, 1.34, 1.50, 1.68, 1.81])
            phi_array = np.array([-0.729, -0.789, -0.910, -1.05, -1.20, -1.26, -1.45, -1.50, -1.49])
            eta_array = np.array([-0.243, -0.261, -0.261, -0.273, -0.278, -0.301, -0.301, -0.319, -0.336])
            alpha = np.interp(logDv, logDv_array, alpha_array)
            beta = np.interp(logDv, logDv_array, beta_array)
            gamma = np.interp(logDv, logDv_array, gamma_array)
            phi = np.interp(logDv, logDv_array, phi_array)
            eta = np.interp(logDv, logDv_array, eta_array)
            if Tinker_z_dep: # Redshift dependence from equations (9-12); only calibrated for M200
                beta *= (1.+z)**0.20
                gamma *= (1.+z)**-0.01
                phi *= (1.+z)**-0.08
                eta *= (1.+z)**0.27
            self.alpha_Tinker = alpha
            self.beta_Tinker = beta
            self.gamma_Tinker = gamma
            self.phi_Tinker = phi
            self.eta_Tinker = eta
            if verbose:
                print('alpha: %1.3f; beta: %1.3f; gamma: %1.3f; phi: %1.3f; eta: %1.3f'%(alpha, beta, gamma, phi, eta))
            # Calibrated halo bias parameters (not from peak-background split) from Table 2
            y = np.log10(self.Dv)
            exp = np.exp(-(4./y)**4)
            self.A_Tinker = 1.+0.24*y*exp
            self.a_Tinker = 0.44*y-0.88
            self.B_Tinker = 0.183
            self.b_Tinker = 1.5
            self.C_Tinker = 0.019+0.107*y+0.19*exp
            self.c_Tinker = 2.4
            if verbose:
                print('A: %1.3f; a: %1.3f'%(self.A_Tinker, self.a_Tinker))
                print('B: %1.3f; b: %1.3f'%(self.B_Tinker, self.b_Tinker))
                print('C: %1.3f; c: %1.3f'%(self.C_Tinker, self.c_Tinker))
        else:
            raise ValueError('Halo model not recognised')
        if verbose:
            print()

    def __str__(self):
        print('Halo model:', self.name)
        print('Scale factor: %1.3f'%(self.a))
        print('Redshift: %1.3f'%(self.z))
        print('Omega_m(z=0): %1.3f'%(self.Om_m))
        print('delta_c: %1.4f'%(self.dc))
        print('Delta_v: %4.1f'%(self.Dv))
        print('Parameters:')
        if self.name in ['Press & Schecter (1974']:
            print('None')
        elif self.name in ['Sheth & Tormen (1999)', 'Sheth, Mo & Tormen (2001)']:
            print('p: %1.3f; q: %1.3f; A: %1.4f'%(self.p_ST, self.q_ST, self.A_ST))
            if self.name in ['Sheth, Mo & Tormen (2001)']:
                print('a: %1.3f; b: %1.3f; c: %1.3f'%(self.a_SMT, self.b_SMT, self.c_SMT))
        elif self.name in ['Despali et al. (2016)']:
            print('p: %1.3f; q: %1.3f; A: %1.4f'%(self.p, self.q, self.A_ST))
        elif self.name in ['Tinker et al. (2010)']:
            print('alpha: %1.3f; beta: %1.3f; gamma: %1.3f; phi: %1.3f; eta: %1.3f'
                %(self.alpha_Tinker, self.beta_Tinker, self.gamma_Tinker, self.phi_Tinker, self.eta_Tinker))
            print('A: %1.3f; a: %1.3f'%(self.A_Tinker, self.a_Tinker))
            print('B: %1.3f; b: %1.3f'%(self.B_Tinker, self.b_Tinker))
            print('C: %1.3f; c: %1.3f'%(self.C_Tinker, self.c_Tinker))
        return ''


    def _mass_function_nu(self, nu:np.ndarray) -> np.ndarray:
        '''
        Halo mass function g(nu) with nu=delta_c/sigma(M)
        Integral of g(nu) over all nu is unity\n
        Args:
            nu: Peak height
        '''
        if self.name == 'Press & Schecter (1974)':
            return np.sqrt(2./np.pi)*np.exp(-(nu**2)/2.)
        elif self.name in ['Sheth & Tormen (1999)', 'Sheth, Mo & Tormen (2001)', 'Despali et al. (2016)']:
            A = self.A_ST; q = self.q_ST; p = self.p_ST
            return A*(1.+((q*nu**2)**(-p)))*np.exp(-q*nu**2/2.)
        elif self.name == 'Tinker et al. (2010)':
            alpha = self.alpha_Tinker
            beta = self.beta_Tinker
            gamma = self.gamma_Tinker
            phi = self.phi_Tinker
            eta = self.eta_Tinker
            f1 = (1.+(beta*nu)**(-2.*phi))
            f2 = nu**(2.*eta)
            f3 = np.exp(-gamma*nu**2/2.)
            return alpha*f1*f2*f3
        else:
            raise ValueError('Halo model not recognised in mass_function')


    def _linear_bias_nu(self, nu:np.ndarray) -> np.ndarray:
        '''
        Halo linear bias b(nu) with nu=delta_c/sigma(M)\n
        Integral of b(nu)*g(nu) over all nu is unity\n
        Args:
            nu: Peak height
        '''
        if self.name == 'Press & Schecter (1974)':
            return 1.+(nu**2-1.)/self.dc
        elif self.name in ['Sheth & Tormen (1999)', 'Despali et al. (2016)']:
            p = self.p_ST; q = self.q_ST
            return 1.+(q*(nu**2)-1.+2.*p/(1.+(q*nu**2)**p))/self.dc
        elif self.name == 'Sheth, Mo & Tormen (2001)':
            a = self.a_SMT; b = self.b_SMT; c = self.c_SMT
            anu2 = a*nu**2
            f1 = np.sqrt(a)*anu2
            f2 = np.sqrt(a)*b*anu2**(1.-c)
            f3 = anu2**c
            f4 = anu2**c+b*(1.-c)*(1.-c/2.)
            return 1.+(f1+f2-f3/f4)/(self.dc*np.sqrt(a))
        elif self.name == 'Tinker et al. (2010)':
            if Tinker_PBS:
                beta = self.beta_Tinker; gamma = self.gamma_Tinker
                phi = self.phi_Tinker; eta = self.eta_Tinker
                f1 = (gamma*nu**2-(1.+2.*eta))/self.dc
                f2 = (2.*phi/self.dc)/(1.+(beta*nu)**(2.*phi))
                return 1.+f1+f2
            else:
                A = self.A_Tinker; a = self.a_Tinker
                B = self.B_Tinker; b = self.b_Tinker
                C = self.C_Tinker; c = self.c_Tinker
                fA = A*nu**a/(nu**a+self.dc**a)
                fB = B*nu**b
                fC = C*nu**c
                return 1.-fA+fB+fC
        else:
            raise ValueError('Halo model not recognised in linear_bias')


    def mass_function(self, M:np.ndarray, sigmas=None, sigma=None, Pk_lin=None) -> np.ndarray:
        '''
        Calculates the halo mass function as a function of halo mass\n
        This is the comoving number-density of haloes per halo mass\n
        Sometimes referred to as n(M) or dn/dM in the literature\n
        Args:
            M: Array of halo masses [Msun/h]
        '''
        F = self.multiplicity_function(M, sigmas, sigma, Pk_lin)
        return F*self.rhom/M**2


    def multiplicity_function(self, M:np.ndarray, sigmas=None, sigma=None, Pk_lin=None) -> np.ndarray:
        '''
        Calculates M^2 n(M) / rhobar, the so-called halo multiplicity function\n
        Note that this is dimensionless\n
        Args:
            M: Array of halo masses [Msun/h]
        '''
        nu = self._peak_height(M, sigmas, sigma, Pk_lin)
        R = cosmology.Lagrangian_radius(M, self.Om_m)
        if Pk_lin is not None:
            dlnsigma2_dlnR = cosmology.dlnsigma2_dlnR(R, Pk_lin)
        elif sigma is not None:
            eps = eps_deriv_mf; dR = R*eps # Uses numerical derivative
            dlnsigma2_dlnR = 2.*util.log_derivative(sigma, R, dR)
        else:
            # TODO: Add calculation of dnu_dlnm for sigmas
            raise ValueError('Error, this currently only works with either P(k) or sigma(R) functions')
        dnu_dlnm = -(nu/6.)*dlnsigma2_dlnR
        return self._mass_function_nu(nu)*dnu_dlnm


    def linear_bias(self, M:np.ndarray, sigmas=None, sigma=None, Pk_lin=None) -> np.ndarray:
        '''
        Calculates the linear halo bias as a function of halo mass\n
        Args:
            M: Array of halo masses [Msun/h]
        '''
        nu = self._peak_height(M, sigmas, sigma, Pk_lin)
        return self._linear_bias_nu(nu)


    def Lagrangian_radius(self, M:np.ndarray) -> np.ndarray:
        '''
        Radius [Mpc/h] of a sphere containing mass M in a homogeneous universe\n
        Args:
            M: Array of halo masses [Msun/h]
        '''
        return cosmology.Lagrangian_radius(M, self.Om_m)


    def average(self, M:np.ndarray, fs:np.ndarray, sigmas=None, sigma=None, Pk_lin=None) -> np.ndarray:
        '''
        Calculate the mean of some f(M) over halo mass <f>: int f(M)n(M)dM where n(M) = dn/dM in some notations.\n
        Note that the units of n(M) are [(Msun/h)^{-1} (Mpc/h)^{-3}] so the units of the result are [F (Mpc/h)^{-3}].\n
        Common use cases:
            <M/rho> = 1 over all halo mass (equivalent to int g(nu)dnu = 1)
            <M^2/rho> = M_NL non-linear halo mass that maximall contributes to one-halo term (not M*)
            <b(M)M/rho> = 1 over all halo mass (equivalent to int g(nu)b(nu)dnu = 1)
            <N(M)> with N the number of galaxies in each halo of mass M; gives mean number density of galaxies
            <b(M)N(M)>/<N(M)> with N the number of galaxies in each halo of mass M; gives mean bias of galaxies\n
        Args:
            M: Array of halo masses [Msun/h]
            fs(M): Array of function to calculate mean density of (same length as M)
            sigmas(M): Optional array of previously calculated nu values corresponding to M
            sigma(R): Optional function to get sigma(R) at z of interest
            Pk_lin(k): Optional function to get linear power at z of interest
        '''
        nu = self._peak_height(M, sigmas, sigma, Pk_lin)
        integrand = (fs/M)*self._mass_function_nu(nu)
        return halo_integration(integrand, nu)*self.rhom


    def power_spectrum(self, k:np.ndarray, M:np.ndarray, profiles:dict, Pk_lin, beta=None, sigmas=None, 
        sigma=None, subtract_shotnoise=True, correct_discrete=True, k_trunc=None, verbose=False) -> tuple:
        '''
        Computes power spectra given that halo model. Returns the two-halo, one-halo and sum terms.\n
        TODO: Remove Pk_lin dependence?\n
        Inputs
            k: Array of wavenumbers [h/Mpc]
            M: Array of halo masses [Msun/h]
            profiles: Dictionary of halo profiles from halo_profile class and corresponding field names
            Pk_lin(k): Function to evaluate the linear power spectrum [(Mpc/h)^3]
            beta(M1, M2, k): Optional array of beta_NL values at points M, M, k
            sigmas(M): Optional pre-computed array of linear sigma(M) values corresponding to M
            sigma(R): Optional function to evaluate the linear sigma(R)
            subtract_shotnoise: Should shot noise be subtracted from discrete spectra?
            correct_discrete: Properly treat discrete tracers with <N(N-1)> rather than <N^2>?
            k_trunc: None or wavenumber [h/Mpc] at which to truncate the one-halo term at large scales
            verbose: verbosity
        '''
        from time import time
        t1 = time() # Initial time

        # Checks
        if not isinstance(profiles, dict): raise TypeError('profiles must be a dictionary')
        for profile in profiles.values():
            if (k != profile.k).all(): raise ValueError('k arrays must all be identical to those in profiles')
            if (M != profile.M).all(): raise ValueError('Mass arrays must be identical to those in profiles')

        # Create arrays of R (Lagrangian radius) and nu values that correspond to the halo mass
        nu = self._peak_height(M, sigmas, sigma, Pk_lin)

        # Calculate the missing halo-bias from the low-mass part of the integral
        A = 1.-integrate.quad(lambda nu: self._mass_function_nu(nu)*self._linear_bias_nu(nu), nu[0], np.inf)[0]
        if verbose: print('Missing halo-bias-mass from the low-mass end of the two-halo integrand:', A)
        if A < 0.:  warnings.warn('Warning: Mass function/bias correction is negative!', RuntimeWarning)

        # Shot noise calculations
        Pk_SN = {}
        for name, profile in profiles.items():
            Pk_SN[name] = self._Pk_1h(M, nu, profile.amp/profile.norm**2) if profile.discrete_tracer else 0.

        # Fill arrays for results
        Pk_2h_dict, Pk_1h_dict, Pk_hm_dict = {}, {}, {}
        Pk_2h, Pk_1h = np.zeros_like(k), np.zeros_like(k)

        # Loop over halo profiles/fields
        for iu, (name_u, profile_u) in enumerate(profiles.items()): 
            for iv, (name_v, profile_v) in enumerate(profiles.items()):
                power_name = name_u+'-'+name_v # Name for this two-point combination
                reverse_name = name_v+'-'+name_u # Reverse name to avoid double computations for cross terms
                if verbose: print('Calculating power:', power_name)
                if iu <= iv:
                    for ik, _k in enumerate(k): # Loop over wavenumbers

                        # Two-halo term, treat non-linear halo bias carefully
                        if beta is None: 
                            Pk_2h[ik] = self._Pk_2h(Pk_lin, _k, M, nu, 
                                                    profile_u.Wk[ik, :], profile_v.Wk[ik, :],
                                                    profile_u.mass_tracer, profile_v.mass_tracer, A)
                        else:
                            Pk_2h[ik] = self._Pk_2h(Pk_lin, _k, M, nu, 
                                                    profile_u.Wk[ik, :], profile_v.Wk[ik, :],
                                                    profile_u.mass_tracer, profile_v.mass_tracer, A, beta[:, :, ik])

                        # One-halo term, treat discrete auto case carefully
                        if name_u == name_v: 
                            if correct_discrete and profile_u.discrete_tracer: # Treat discrete tracers
                                Wfac = profile_u.amp*(profile_u.amp-1.) # <N(N-1)> for discrete tracers
                            else:
                                Wfac = profile_u.amp**2 # <N^2> for others
                            if profile_u.var is not None: Wfac += profile_u.var # Add variance
                            Wprod = Wfac*(profile_u.Uk[ik, :]/profile_u.norm)**2 # Multiply by factors of normalisataion and profile
                        else:
                            Wprod = profile_u.Wk[ik, :]*profile_v.Wk[ik, :]
                        Pk_1h[ik] = self._Pk_1h(M, nu, Wprod)

                    # Shot noise corrections
                    # If '(not discrete) and shot' or 'discrete and (not shot)' no need to do anything as shot noise already correct
                    if (name_u == name_v) and profile_u.discrete_tracer:
                        if correct_discrete and (not subtract_shotnoise):
                            Pk_1h += Pk_SN[name_u] # Need to add shot noise
                        elif (not correct_discrete) and subtract_shotnoise:
                            warnings.warn('Warning: Subtracting shot noise while not treating discreteness properly is dangerous', RuntimeWarning)
                            Pk_1h[:] -= Pk_SN[name_u] # Need to subtract shot noise
                    if k_trunc is not None: Pk_1h *= 1.-np.exp(-(k/k_trunc)**2) # Suppress one-halo term

                    # Finish
                    Pk_2h_dict[power_name], Pk_1h_dict[power_name] = Pk_2h.copy(), Pk_1h.copy()
                    Pk_hm_dict[power_name] = Pk_2h_dict[power_name]+Pk_1h_dict[power_name]

                else:
                    # No need to do these calculations twice
                    Pk_2h_dict[power_name] = Pk_2h_dict[reverse_name]
                    Pk_1h_dict[power_name] = Pk_1h_dict[reverse_name]
                    Pk_hm_dict[power_name] = Pk_hm_dict[reverse_name]

        # Finish
        t2 = time() # Final time
        if verbose: print('Halomodel calculation time [s]:', t2-t1, '\n')
        return Pk_2h_dict, Pk_1h_dict, Pk_hm_dict


    def _Pk_2h(self, Pk_lin, k:float, M:np.ndarray, nu:np.ndarray, 
        Wu:float, Wv:float, mass_u:bool, mass_v:bool, A:float, beta=None) -> float:
        '''
        Two-halo term at a specific wavenumber\n
        '''
        if beta is None:
            I_NL = 0.
        else:
            I_NL = self._I_beta(beta, M, nu, Wu, Wv, mass_u, mass_v, A)
        Iu = self._I_2h(M, nu, Wu, mass_u, A)
        Iv = self._I_2h(M, nu, Wv, mass_v, A)
        return Pk_lin(k)*(Iu*Iv+I_NL)


    def _Pk_1h(self, M:np.ndarray, nu:np.ndarray, WuWv:float) -> float:
        '''
        One-halo term at a specific wavenumber\n
        '''
        integrand = WuWv*self._mass_function_nu(nu)/M
        P_1h = halo_integration(integrand, nu)
        P_1h = P_1h*self.rhom
        return P_1h


    def _I_2h(self, M:np.ndarray, nu:np.ndarray, W:float, mass:bool, A:float) -> float:
        '''
        Evaluate the integral that appears in the two-halo term\n
        '''
        integrand = W*self._linear_bias_nu(nu)*self._mass_function_nu(nu)/M
        I_2h = halo_integration(integrand, nu)
        if mass: I_2h += A*W[0]/M[0]
        I_2h = I_2h*self.rhom
        return I_2h


    def _I_beta(self, beta:np.ndarray, M:np.ndarray, nu:np.ndarray, Wu:float, Wv:float, 
        massu:bool, massv:bool, A:float) -> float:
        '''
        Evaluates the beta_NL double integral\n
        TODO: Loops maybe horribly inefficient loops here (outer product?)
        '''
        from numpy import trapz
        integrand = np.zeros((len(nu), len(nu)))
        for iM1, nu1 in enumerate(nu):
            for iM2, nu2 in enumerate(nu):
                if iM2 >= iM1:
                    M1, M2 = M[iM1], M[iM2]
                    W1, W2 = Wu[iM1], Wv[iM2]
                    g1 = self._mass_function_nu(nu1)
                    g2 = self._mass_function_nu(nu2)
                    b1 = self._linear_bias_nu(nu1)
                    b2 = self._linear_bias_nu(nu2)
                    integrand[iM1, iM2] = beta[iM1, iM2]*W1*W2*g1*g2*b1*b2/(M1*M2)
                else:
                    integrand[iM1, iM2] = integrand[iM2, iM1]
        integral = util.trapz2d(integrand, nu, nu)
        if do_I11 and massu and massv:
            integral += beta[0, 0]*(A**2)*Wu[0]*Wv[0]/M[0]**2
        if do_I12_I21 and massu:
            # integrand = np.zeros(len(nu))
            # for iM, nu in enumerate(nu):
            #     M, W = M[iM], Wv[iM]
            #     g = self._mass_function_nu(nu)
            #     b = self._linear_bias_nu(nu)
            #     integrand[iM] = beta[0, iM]*W*g*b/M
            g = self._mass_function_nu(nu)
            b = self._linear_bias_nu(nu)
            integrand = beta[0, :]*Wv*g*b/M
            integral += (A*Wu[0]/M[0])*trapz(integrand, nu)
        if do_I12_I21 and massv:
            # integrand = np.zeros(len(nu))
            # for iM, nu in enumerate(nu):
            #     M, W = M[iM], Wu[iM]
            #     g = self._mass_function_nu(nu)
            #     b = self._linear_bias_nu(nu)
            #     integrand[iM] = beta[iM, 0]*W*g*b/M
            g = self._mass_function_nu(nu)
            b = self._linear_bias_nu(nu)
            integrand = beta[:, 0]*Wu*g*b/M
            integral += (A*Wv[0]/M[0])*trapz(integrand, nu)
        return integral*self.rhom**2


    def power_spectrum_cross_halo_mass(self, k:np.ndarray, M:np.ndarray, profiles:dict, M_halo:float,
        Pk_lin, beta=None, sigmas=None, sigma=None, verbose=True) -> tuple:
        '''
        Computes the power spectrum of a single tracer crossed with haloes of a single specific mass\n
        Args:
            k: Array of wavenumbers [h/Mpc]
            M: Array of halo masses [Msun/h]
            profiles: A dictionary of halo profiles (halo_profile class) with associated field names
            M_halo: Halo mass at which to evaluatae cross spectra [Msun/h]
            Pk_lin(k): Function to evaluate the linear power spectrum [(Mpc/h)^3]
            beta(M, k): Optional array of beta_NL values at points M, k
            sigmas(M): Optional pre-computed array of linear sigma(M) values corresponding to M
            sigma(R): Optional function to evaluate the linear sigma(R)
            verbose: verbosity
        '''
        from time import time
        from scipy.interpolate import interp1d
        t1 = time() # Initial time

        # Checks
        #if type(profiles) != dict: raise TypeError('profiles must be supplied as a dictionary')
        if not isinstance(profiles, dict): raise TypeError('profiles must be a dictionary')
        for profile in profiles:
            if (k != profile.k).all(): raise ValueError('k arrays must all be identical to those in profiles')
            if (M != profile.M).all(): raise ValueError('Mass arrays must be identical to those in profiles')

        # Create arrays of R (Lagrangian radius) and nu values that correspond to the halo mass
        nu = self._peak_height(M, sigmas, sigma, Pk_lin)

        # Calculate the missing halo-bias from the low-mass part of the integral
        integrand = self._mass_function_nu(nu)*self._linear_bias_nu(nu)
        A = 1.-halo_integration(integrand, nu)
        if verbose: print('Missing halo-bias-mass from two-halo integrand:', A, '\n')

        # Calculate nu(Mh) and W(Mh, k) by interpolating the input arrays
        # NOTE: Wk_halo_mass is not the halo profile, but the profile of the field evaluated at the halo mass!
        nu_M_interp = interp1d(np.log(M), nu, kind='cubic')
        nu_halo = nu_M_interp(np.log(M_halo))
        Wk_halo_mass = {}
        for name, profile in profiles.items():
            Wk = np.empty_like(profile.Wk[:, 0])
            for ik, _ in enumerate(k):
                WM_interp = interp1d(np.log(M), profile.Wk[ik, :], kind='cubic')
                Wk[ik] = WM_interp(np.log(M_halo))
            Wk_halo_mass[name] = Wk.copy()

        # Combine everything and return
        Pk_2h_dict, Pk_1h_dict, Pk_hm_dict = {}, {}, {}
        Pk_2h, Pk_1h = np.zeros_like(k), np.zeros_like(k)
        for name, profile in profiles.items():
            for ik, _k in enumerate(k):
                if beta is None:
                    Pk_2h[ik] = self._Pk_2h_hu(Pk_lin, _k, M, nu_halo, nu, profile.Wk[ik, :], profile.mass_tracer, A)
                else:
                    Pk_2h[ik] = self._Pk_2h_hu(Pk_lin, _k, M, nu_halo, nu, profile.Wk[ik, :], profile.mass_tracer, A, beta[:, ik])
                Pk_1h[ik] = Wk_halo_mass[name][ik] # Simply the halo profile at M=Mh here
                Pk_2h_dict[name], Pk_1h_dict[name] = Pk_2h.copy(), Pk_1h.copy()
                Pk_hm_dict[name] = Pk_2h_dict[name]+Pk_1h_dict[name]
        t2 = time() # Final time

        if verbose:  
            print('Halomodel calculation time [s]:', t2-t1, '\n')

        return Pk_2h_dict, Pk_1h_dict, Pk_hm_dict


    def _Pk_2h_hu(self, Pk_lin, k:float, M:np.ndarray, nuh:float, 
        nu:np.ndarray, Wk:float, mass:bool, A:float, beta=None) -> float:
        '''
        Two-halo term for halo-u at a specific wavenumber\n
        '''
        I_NL = self._I_beta_hu(beta, M, nuh, nu, Wk, mass, A) if beta is not None else 0.
        Ih = self._linear_bias_nu(nuh) # Simply the linear bias
        Iu = self._I_2h(M, nu, Wk, mass, A) # Same as for the standard two-halo term
        return Pk_lin(k)*(Ih*Iu+I_NL)


    def _I_beta_hu(self, beta:np.ndarray, M:np.ndarray, nu_halo:float, nu:np.ndarray, 
        Wk:np.ndarray, mass:bool, A:float) -> float:
        '''
        Evaluates the beta_NL integral for halo-u\n
        '''
        from numpy import trapz
        b_halo = self._linear_bias_nu(nu_halo)
        g = self._mass_function_nu(nu)
        b = self._linear_bias_nu(nu)
        integrand = beta*Wk*g*b/M
        integral = trapz(integrand, nu)
        if mass: integral += A*beta[0]*Wk[0]/M[0]
        return b_halo*integral*self.rhom


    def _peak_height(self, M:np.ndarray, sigmas=None, sigma=None, Pk_lin=None) -> np.ndarray:
        '''
        Calculate peak-height (nu) values from array of halo masses\n
        '''
        # Create arrays of R (Lagrangian) and nu values that correspond to the halo mass
        R = cosmology.Lagrangian_radius(M, self.Om_m)

        # Convert R values to nu via sigma(R)
        if sigmas is not None:
            nu = self.dc/sigmas # Use the provided sigma(R) values or...
        elif sigma is not None:
            nu = self.dc/sigma(R) # ...otherwise evaluate the provided sigma(R) function or...
        elif Pk_lin is not None:
            nu = self.dc/cosmology.sigmaR(R, Pk_lin, integration_type='quad') # ...otherwise integrate
        else:
            raise ValueError('Error, you need to specify (at least) one of Pk_lin, sigma or sigmas') 
        return nu


    def virial_radius(self, M:np.ndarray) -> np.ndarray:
        '''
        Halo virial radius based on the halo mass and overdensity condition\n
        Args:
            M: Halo mass [Msun/h]
        '''
        return cosmology.Lagrangian_radius(M, self.Om_m)/np.cbrt(self.Dv)

### ###

### Beta_NL ###

def interpolate_beta_NL(k:np.ndarray, M:np.ndarray, M_small:np.ndarray, beta_NL_small:np.ndarray, 
    scheme='RegularGridInterp', **kwargs) -> np.ndarray:
    '''
    Wrapper for various beta_NL interpolation schemes to go from coarse grid of halo masses\n
    to a finer grid of halo masses.\n
    Args:
        k: Array of wavenumbers [h/Mpc]
        M: Array of desired halo masses [Msun/h]
        M_small: Array of halo masses at which beta_NL_small is evaluated already
        beta_NL_small: Array beta_NL(M_small, M_small, k)
        scheme: Interpolation scheme
        **kwargs: To be passed to interpolation schemes
    '''
    if scheme == 'interp2d':
        beta_NL = _interpolate_beta_NL(k, M, M_small, beta_NL_small, **kwargs)
    elif scheme == 'RegularGridInterp':
        beta_NL = _RegularGridInterp_beta_NL(k, M_small, beta_NL_small, k, M, **kwargs)
    elif scheme == 'RegularGridInterp_log':
        beta_NL = _RegularGridInterp_beta_NL_log(k, M_small, beta_NL_small, k, M, **kwargs)
    else:
        raise ValueError('beta_NL interpolation method not recognised')
    return beta_NL


def _interpolate_beta_NL(k:np.ndarray, M:np.ndarray, M_small:np.ndarray, beta_NL_small:np.ndarray, 
    fill_value=0.) -> np.ndarray:
    '''
    Interpolate beta_NL from a small grid to a large grid for halo-model calculations.\n
    TODO: Remove inefficient loops?\n
    TODO: Add more interesting extrapolations. Currently just nearest neighbour,\n
    which means everything outside the range is set to the same value.\n
    '''
    from scipy.interpolate import interp2d
    beta_NL = np.zeros((len(M), len(M), len(k))) # Array for output
    for ik, _ in enumerate(k):
        beta_NL_interp = interp2d(np.log(M_small), np.log(M_small), beta_NL_small[:, :, ik],
            kind='linear', fill_value=fill_value)
        for iM1, M1 in enumerate(M):
            for iM2, M2 in enumerate(M):
                beta_NL[iM1, iM2, ik] = beta_NL_interp(np.log(M1), np.log(M2))
    return beta_NL


def _RegularGridInterp_beta_NL(k_in:np.ndarray, M_in:np.ndarray, beta_NL_in:np.ndarray, 
    k_out:np.ndarray, M_out:np.ndarray, method='linear') -> np.ndarray:
    '''
    Supported methods are 'linear', 'nearest', 'slinear', 'cubic', and 'quintic'\n
    '''
    from scipy.interpolate import RegularGridInterpolator
    bnl_interp = RegularGridInterpolator([M_in, M_in, np.log(k_in)], beta_NL_in,
        method=method, fill_value=None, bounds_error=False)
    bnl_out = np.zeros((M_out.size, M_out.size, k_out.size))
    indices = np.vstack(np.meshgrid(np.arange(M_out.size), np.arange(M_out.size), np.arange(k_out.size), 
                                    copy=False)).reshape(3, -1).T
    values = np.vstack(np.meshgrid(M_out, M_out, np.log(k_out), 
                                   copy=False)).reshape(3, -1).T
    bnl_out[indices[:, 0], indices[:, 1], indices[:, 2]] = bnl_interp(values)
    return bnl_out


def _RegularGridInterp_beta_NL_log(k_in:np.ndarray, M_in:np.ndarray, beta_NL_in:np.ndarray, 
    k_out:np.ndarray, M_out:np.ndarray, method='linear') -> np.ndarray:
    '''
    Supported methods are 'linear', 'nearest', 'slinear', 'cubic', and 'quintic'\n
    '''
    from scipy.interpolate import RegularGridInterpolator
    bnl_interp = RegularGridInterpolator([np.log10(M_in), np.log10(M_in), np.log(k_in)], beta_NL_in,
        method=method, fill_value=None, bounds_error=False)
    bnl_out = np.zeros((M_out.size, M_out.size, k_out.size))
    indices = np.vstack(np.meshgrid(np.arange(M_out.size), np.arange(M_out.size), np.arange(k_out.size), 
                                    copy=False)).reshape(3, -1).T
    values = np.vstack(np.meshgrid(np.log10(M_out), np.log10(M_out), np.log(k_out), 
                                   copy=False)).reshape(3, -1).T
    bnl_out[indices[:, 0], indices[:, 1], indices[:, 2]] = bnl_interp(values)
    return bnl_out

### ###

### Halo profiles ###

class profile():
    '''
    Class for halo profiles\n
    '''
    def __init__(self, k:np.ndarray, M:np.ndarray, Uk:np.ndarray, Wk:np.ndarray, 
        amp=None, norm=1., var=None, mass_tracer=False, discrete_tracer=False):
        # TODO: Is .copy() necessaary here?
        self.k, self.M = k.copy(), M.copy()
        self.Uk, self.Wk = Uk.copy(), Wk.copy()
        self.amp = amp.copy() if amp is not None else None
        self.norm = norm
        self.var = var.copy() if var is not None else None
        self.mass_tracer, self.discrete_tracer = mass_tracer, discrete_tracer

    def __str__(self):
        log_limit = 1e5
        print('Halo profile')
        print('Mass tracer:', self.mass_tracer)
        print('Discrete tracer:', self.discrete_tracer)
        if self.norm != 1.: print('Field normalisation:', self.norm)
        print('Number of wavenumber bins:', len(self.k))
        print('Number of mass bins:', len(self.M))
        print('Wavenumber range [log10(h/Mpc)]: %1.3f %1.3f'%(np.log10(self.k[0]), np.log10(self.k[-1])))
        print('Halo mass range [log10(Msun/h)]: %1.3f %1.3f'%(np.log10(self.M[0]), np.log10(self.M[-1])))
        print('The following are at the low and high halo mass ends')
        if self.amp[0] > log_limit:
            print('Profile amplitude mean [log10]:', np.log10(self.amp[0]), np.log10(self.amp[-1]))
        else:
            print('Profile amplitude mean:', self.amp[0], self.amp[-1])
        if self.var is not None:
            if self.var[0] > log_limit:
                print('Profile amplitude variance [log10]:', np.log10(self.var[0]), np.log10(self.var[-1]))
            else:
                print('Profile amplitude variance:', self.var[0], self.var[-1])
        print('Dimensionless profiles at low k (should be ~1):', self.Uk[0, 0], self.Uk[0, -1])
        print('Dimensionful profiles at low k (should be amplitudes):', self.Wk[0, 0], self.Wk[0, -1])
        return ''

    @classmethod
    def Fourier(cls, k:np.ndarray, M:np.ndarray, Uk:np.ndarray,
        amp=None, norm=1., var=None, mass_tracer=False, discrete_tracer=False):
        '''
        Class initialisation for Fourier space halo profiles.\n
        Uk is assumed to be an array Uk(k, M) of the profile Fourier transform.\n
        If amp=None then Uk is assumed to be dimensionful, otherwise Uk is assumed dimensionless with Uk(k->0, M) = 1\n
        Matter profiles have amp=M (halo mass), galaxy profiles have amp=N (number of galaxies)\n
        Args:
            k: Array of wavenumbers [h/Mpc] going from low to high
            M: Array of halo masses [Msun/h] going from low to high
            Uk: 2D array of halo Fourier transform (order k, M)
            amp: 1D array of halo profile amplitudes at halo masses 'M' (e.g., M for mass; N for galaxies)
            norm: Float for field normalisation (e.g., rhom for mass, ng for galaxies)
            var: Variance in the profile amplitude at each halo mass (e.g., N for Poisson galaxies)
            mass_tracer: Are contributions expected for M < M[0] (e.g., matter)?
            discrete_tracer: Does the profile correspond to that of a discrete tracer (e.g., galaxies)?
        '''
        # Set internal variables
        if Uk.shape != (k.shape[0], M.shape[0]): raise ValueError('Array shapes do not match')
        #cls.k, cls.M = k.copy(), M.copy()
        #cls.mass_tracer, cls.discrete_tracer = mass_tracer, discrete_tracer
        #cls.norm = norm
        #cls.var = var.copy() if var is not None else var
        if amp is None:
            amp = Uk[0, :] # TODO: Integrate to get this? This relies on k[0] being small!
            Wk = Uk
            _Uk = Uk/amp
        else:
            #cls.amp = amp.copy()
            #cls.Uk = Uk.copy()
            #self.Wk = (self.Uk.T*self.amp).T/self.norm # Transposes necessary to get multiplication correct
            _Uk = Uk
            Wk = (amp*Uk)/norm
            
        return cls(k, M, _Uk, Wk, amp=amp, norm=norm, var=var, mass_tracer=mass_tracer, discrete_tracer=discrete_tracer)

    @classmethod
    def configuration(cls, k:np.ndarray, M:np.ndarray, Prho, rv:np.ndarray, c:np.ndarray,
        amp=None, norm=1., var=None, mass_tracer=False, discrete_tracer=False):
        ''''
        Alternative class initialisation for configuration-space haloes.\n
        Prho is a halo (density or whatever) profile multiplied by 4pi r^2.\n
        If amp=None then Prho is assumed to be normalised correctly, otherwise Phro can\n
        have any normalisation and will be renormalaised by amp to have the correct 'mass'.\n
        Args:
            k: Array of wavenumbers [h/Mpc] going from low to high
            M: Array of halo masses [Msun/h] going from low to high
            Prho: Function to evaluate halo profile as a function of radius Prho(r, M, rv, c)
            rv: Array of halo virial radii [Mpc/h]
            c: Array of halo concentration
            amp: Array of halo profile amplitudes at halo masses 'M' (e.g., M for mass; N for galaxies)
            norm: Float for field normalisation (e.g., rhom for mass, ng for galaxies)
            var: Var(N(M)) (auto)variance in the profile amplitude at each halo mass (e.g., N for Poisson galaxies)
            mass_tracer: Are contributions expected for M < M[0] (e.g., matter)?
            discrete_tracer: Does the profile correspond to that of a discrete tracer (e.g., galaxies)?
        '''
        _amp = np.zeros_like(M)
        Uk, Wk = np.zeros((len(k), len(M))), np.zeros((len(k), len(M)))
        for iM, (_M, _rv, _c) in enumerate(zip(M, rv, c)):
            _amp[iM] = _halo_window(0., _M, _rv, _c, Prho) # Amplitiude
        for ik, _k in enumerate(k):
            for iM, (_M, _rv, _c) in enumerate(zip(M, rv, c)):
                W = _halo_window(_k, _M, _rv, _c, Prho)
                if amp is None:
                    Wk[ik, iM] = W
                    Uk[ik, iM]= Wk[ik, iM]/_amp[iM] # Normalise
                else:
                    Uk[ik, iM] = W/_amp[iM]
                    Wk[ik, iM] = Uk[ik, iM]*amp[iM]
        if amp is not None: _amp = amp
        profile = cls.Fourier(k, M, Uk, amp=_amp, norm=norm, var=var, mass_tracer=mass_tracer, discrete_tracer=discrete_tracer)
        return profile

### Halo profiles in configuration space ###

def _halo_window(k:float, M:float, rv:float, c:float, Prho:callable) -> float:
    '''
    Compute the halo window function via integration given a 'density' profile Prho(r) = 4*pi*r^2*rho(r)\n
    TODO: This should almost certainly be done with a dedicated integration routine (FFTlog?)\n
    TODO: Can this be made to accept arrays of M, rv, c as arguments?
    Args:
        k: Fourier wavenumber [h/Mpc]
        M: Halo mass [Msun/h]
        rv: Halo virial radius [Mpc/h]
        c: Halo concentration
        Prho(r, M, rv, c): Function 4pi*r^2*rho(M, r) values at different radii
    '''
    from scipy.special import spherical_jn

    if win_integration in [integrate.trapezoid, integrate.simps, integrate.romb]:
        R = np.linspace(0., rv, nr_win_integration)
        integrand = spherical_jn(0, k*R)*Prho(R, M, rv, c)
        if win_integration in [integrate.romb]: dR = R[1]-R[0]

    # Integration
    if win_integration in [integrate.trapezoid, integrate.simps]:
        Wk = win_integration(integrand, R)
    elif win_integration in [integrate.romb]:
        Wk = win_integration(integrand, dR)
    elif win_integration in [integrate.quad]:
        Wk, _ = win_integration(lambda r: spherical_jn(0, k*r)*Prho(r, M, rv, c), 0., rv, epsrel=eps_win_integration)
    elif win_integration in [integrate.quadrature]:
        Wk, _ = win_integration(lambda r: spherical_jn(0, k*r)*Prho(r, M, rv, c), 0., rv, rtol=eps_win_integration)
    elif win_integration in [integrate.romberg]:
        Wk = win_integration(lambda r: spherical_jn(0, k*r)*Prho(r, M, rv, c), 0., rv, rtol=eps_win_integration)
    else:
        raise ValueError('Halo window function integration method not recognised')
    return Wk


def Prho(r:np.ndarray, M:float, rv:float, c:float, name=None) -> np.ndarray:
    '''
    Profile (density/emissivity) of a halo multiplied by 4pir^2\n
    Defined to avoid infinities when evaluated at r=0\n
    Args:
        r: Radius from halo centre [Mpc/h]
        M: Halo mass [Msun/h]
        rv: Halo cirial radius [Mpc/h]
        c: Halo concentration
        name: String of profile name:
    '''
    if name == 'isothermal':
        Prho = _Prho_isothermal(r, M, rv)
    elif name == 'NFW':
        Prho = _Prho_NFW(r, M, rv, c)
    else:
        raise ValueError('Halo profile not recognised')
    return Prho

# def rho(r:np.ndarray, rv:float, M:float, *args, name=None):
#     '''
#     Profile (density/emissivity) of a halo
#     Args:
#         r: Radius from halo centre [Mpc/h]
#         rv: Virial radius [Mpc/h]
#         M: Halo mass [Msun/h]
#         *args: Arguments to be passed to the profile (e.g., concentration)
#         name: String of profile name:
#     '''
#     Prh = Prho(r, rv, M, *args, name=name)
#     rho = _rho_from_Prho(Prh, r, rv, M, *args)
#     return rho

# def _rho_from_Prho(Prho, r, rv, M, *args):
#     '''
#     Converts a Prho profile to a rho profile (factor of 4pi r^2)
#     Take care evaluating this at zero (which will give infinity)
#     '''
#     return Prho(r, rv, M, *args)/(4.*np.pi*r**2)

def _Prho_isothermal(_, M, rv):
    '''
    Isothermal density profile multiplied by 4*pi*r^2\n
    '''
    return M/rv


def _Prho_NFW(r, M, rv, c):
    '''
    NFW density profile multiplied by 4*pi*r^2\n
    '''
    rs = rv/c
    return (M/c)*(r/rv**2)*(c**3)*_NFW_factor(c)/(1.+r/rs)**2


# def Prho_UPP(r, rv, M, r500, z, Om_r, Om_m, Om_w, Om_v, h):
#     '''
#     Universal pressure profile: UPP
#     '''
#     alphap = 0.12
#     def p(x):
#         P0 = 6.41
#         c500 = 1.81
#         alpha = 1.33
#         beta = 4.13
#         gamma = 0.31
#         y = c500*x
#         f1 = y**(2.-gamma)
#         f2 = (1.+y**alpha)**(beta-gamma)/alpha
#         p = P0*(h/0.7)**(-3./2.)*f1*(r500/c500)**2/f2
#         return p
#     a = utility.scale_factor_z(z)
#     H = utility.H(Om_r, Om_m, Om_w, Om_v, a)
#     f1 = 1.65*(h/0.7)**2*H**(8./3.)
#     f2 = (M/2.1e14)**(2./3.+alphap)
#     return f1*f2*p(r/r500)*4.*np.pi

### ###

### Halo profiles in Fourier space ###

def window_function(k:np.ndarray, rv:np.ndarray, *args, profile=None) -> np.ndarray:
    '''
    Normalised Fourier tranform for a delta-function profile (unity)\n
    Args:
        k: Array of Fourier wavenumbers [h/Mpc]
        rv: Array of halo virial radii [Mpc/h]
        *args: Additional arguments for halo profiles (e.g., concentration)
        profile: String delta, isothermal, NFW
    '''
    if profile == 'delta':
        Wk = np.ones((len(k), len(rv)))
    elif profile == 'isothermal':
        Wk = _win_isothermal(k, rv)
    elif profile == 'NFW':
        Wk = _win_NFW(k, rv, *args)
    else:
        raise ValueError('Halo profile not recognised')
    return Wk


def _win_isothermal(k:np.ndarray, rv:np.ndarray) -> np.ndarray:
    '''
    Normalised Fourier transform for an isothermal profile\n
    Args:
        k: Array of Fourier wavenumber [h/Mpc]
        rv: Array of halo virial radius [Mpc/h]
    '''
    from scipy.special import sici
    kv = np.outer(k, rv)
    Si, _ = sici(kv)
    Wk = Si/(kv)
    return Wk


def _win_NFW(k:np.ndarray, rv:np.ndarray, c:np.ndarray) -> np.ndarray:
    '''
    Normalised Fourier transform for an NFW profile\n
    Args:
        k: Fourier wavenumber [h/Mpc]
        rv: Halo virial radius [Mpc/h]
        c: Halo concentration
    '''
    from scipy.special import sici
    rs = rv/c
    kv = np.outer(k, rv)
    ks = np.outer(k, rs)
    Sisv, Cisv = sici(ks+kv)
    Sis, Cis = sici(ks)
    f1 = np.cos(ks)*(Cisv-Cis)
    f2 = np.sin(ks)*(Sisv-Sis)
    f3 = np.sin(kv)/(ks+kv)
    f4 = _NFW_factor(c)
    Wk = (f1+f2-f3)/f4
    return Wk


def _NFW_factor(c:np.ndarray) -> np.ndarray:
    '''
    Factor from normalisation that always appears in NFW equations\n
    Args:
        c: Halo concentration
    '''
    return np.log(1.+c)-c/(1.+c)


def matter_profile(k:np.ndarray, M:np.ndarray, rv:np.ndarray, c:np.ndarray, Om_m:float) -> profile:
    '''
    Pre-configured matter NFW profile\n
    Args:
        k: Array of wavenumbers [h/Mpc]
        M: Array of halo masses [Msun/h]
        rv: Array of halo virial radii [Mpc/h]
        c: Array of halo concentrations
        Om_m: Cosmological matter density
    '''
    rhom = cosmology.comoving_matter_density(Om_m)
    Uk = window_function(k, rv, c, profile='NFW')
    return profile.Fourier(k, M, Uk, M, rhom, mass_tracer=True, discrete_tracer=False)


# def galaxy_profile(ks:np.ndarray, Ms:np.ndarray, rvs:np.ndarray, cs:np.ndarray, rhog:float, 
#     HOD_method='Zheng et al. (2005)') -> halo_profile:
#     '''
#     Pre-configured galaxy profile\n
#     # TODO: Need rhog here, but to get this requires integrating Ns, which would require hmod as argument
#     Args:
#         ks: Array of wavenumbers [h/Mpc]
#         Ms: Array of halo masses [Msun/h]
#         rvs: Array of halo virial radii [Mpc/h]
#         cs: Array of halo concentrations
#         rhog: Galaxy number density []
#         HOD_method: String for HOD choice
#     '''
#     N_cen, N_sat = HOD_mean(Ms, method=HOD_method)
#     N_gal = N_cen+N_sat
#     V_cen, V_sat, _ = HOD_variance(N_cen, N_sat)
#     V_gal = V_cen+V_sat
#     Uk_gal = halo_window_function(ks, rvs, profile='isothermal')
#     #Uk_gal = halo_window_function(ks, rvs, cs, profile='NFW')
#     profile = halo_profile(ks, Ms, N_gal, Uk_gal, rhog, var=V_gal, mass=False, discrete=True)
#     return profile


# def central_satellite_profiles(ks:np.ndarray, Ms:np.ndarray, rvs:np.ndarray, cs:np.ndarray, rhog:float, 
#     HOD_method='Zheng et al. (2005)') -> tuple:
#     '''
#     Pre-configured central and satellite galaxy profiles\n
#     # TODO: Need rhog here, but to get this requires integrating Ns, which would require hmod as argument
#     Args:
#         ks: Array of wavenumbers [h/Mpc]
#         Ms: Array of halo masses [Msun/h]
#         rvs: Array of halo virial radii [Mpc/h]
#         cs: Array of halo concentrations
#         rhog: Galaxy number density []
#         HOD_method: String for HOD choice
#     '''
#     N_cen, N_sat = HOD_mean(Ms, method=HOD_method)
#     V_cen, V_sat, _ = HOD_variance(N_cen, N_sat)
#     Uk_cen = halo_window_function(ks, rvs, profile='delta')
#     Uk_sat = halo_window_function(ks, rvs, profile='isothermal')
#     #Uk_sat = halo_window_function(ks, rvs, cs, profile='NFW')
#     profile_cen = halo_profile(ks, Ms, N_cen, Uk_cen, rhog, var=V_cen, mass=False, discrete=True)
#     profile_sat = halo_profile(ks, Ms, N_sat, Uk_sat, rhog, var=V_sat, mass=False, discrete=True)
#     return profile_cen, profile_sat

### ###

### Halo concentration

def concentration(M:np.ndarray, z:float, method='Duffy et al. (2008)', halo_definition='M200') -> np.ndarray:
    '''
    Halo concentration as a function of halo mass and redshift\n
    Args:
        M: Halo mass [Msun/h]
        z: redshift
        method: Duffy et al. (2008)
        halo_definition: M200, Mvir, M200c
    '''
    if method == 'Duffy et al. (2008)':
        c = _concentration_Duffy(M, z, halo_definition)
    else:
        raise ValueError('Halo concentration not recognised')
    return c

def _concentration_Duffy(M:np.ndarray, z:float, halo_definition='M200') -> np.ndarray:
    '''
    Duffy et al (2008; 0804.2486) c(M) relation for WMAP5, See Table 1\n
    Appropriate for the full (rather than relaxed) samples\n
    Args:
        M: Halo mass [Msun/h]
        z: redshift
        halo_definition: M200, Mvir, M200c
    '''
    M_piv = 2e12 # Pivot mass [Msun/h]
    if halo_definition in ['M200', '200', '200b']:
        A = 10.14; B = -0.081; C = -1.01
    elif halo_definition in ['vir', 'virial', 'Mvir']:
        A = 7.85; B = -0.081; C = -0.71
    elif halo_definition in ['M200c', '200c']:
        A = 5.71; B = -0.084; C = -0.47
    else:
        raise ValueError('Halo definition not recognised')
    return A*(M/M_piv)**B*(1.+z)**C # Equation (4) in 0804.2486, parameters from 10th row of Table 1

### ###

### Halo-occupation distribution (HOD) ###

def HOD_mean(M:float, method='Zheng et al. (2005)', **kwargs) -> tuple:
    '''
    Returns the expected (mean) number of central and satellite galaxies as a tuple\n
    Args:
        M: Halo mass [Msun/h]
        method: simple, Zehavi et al. (2004), Zheng et al. (2005), Zhai et al. (2017)
        **kwargs: For specific HOD
    '''
    if method == 'simple':
        result = _HOD_simple(M, **kwargs)
    elif method == 'Zehavi et al. (2004)':
        result = _HOD_Zehavi(M, **kwargs)
    elif method == 'Zheng et al. (2005)':
        result = _HOD_Zheng(M, **kwargs)
    elif method == 'Zhai et al. (2017)':
        result = _HOD_Zhai(M, **kwargs)
    else:
        raise ValueError('HOD method not recognised')
    return result

def _HOD_simple(M:np.ndarray, Mmin=1e12, Msat=1e13, alpha=1.) -> tuple:
    '''
    Simple HOD model\n
    Args:
        M: Halo mass [Msun/h]
        Mmin: Minimum halo mass to host a central galaxy [Msun/h]
        Mmin: Minimum halo mass to host a satellite galaxy [Msun/h]
        alpha: Slope of satellite occupancy
    '''
    Nc = np.heaviside(M-Mmin, 1.)
    Ns = Nc*(M/Msat)**alpha
    return Nc, Ns


def _HOD_Zehavi(M:np.ndarray, Mmin=1e12, M1=1e13, alpha=1.) -> tuple:
    '''
    HOD model from Zehavi et al. (2004; https://arxiv.org/abs/astro-ph/0703457)\n
    Same as Zheng model in the limit that sigma=0 and M0=0\n
    Mean number of central galaxies is only ever 0 or 1 in this HOD\n
    Args:
        M: Halo mass [Msun/h]
        Mmin: Minimum halo mass to host a central galaxy [Msun/h]
        M1: M1 parameter [Msun/h]
        alpha: Slope of satellite occupancy
    '''
    Nc = np.heaviside(M-Mmin, 1.)
    Ns = (M/M1)**alpha
    return Nc, Ns


def _HOD_Zheng(M:np.ndarray, Mmin=1e12, sigma=0.15, M0=1e12, M1=1e13, alpha=1.) -> tuple:
    '''
    Zheng et al. (2005; https://arxiv.org/abs/astro-ph/0408564) HOD model\n
    Args:
        M: Halo mass [Msun/h]
        sigma: Width of central galaxy occupancy
        Mmin: Minimum halo mass to host a central galaxy [Msun/h]
        M0: M0 parameter [Msun/h]
        M1: M1 parameter [Msun/h]
        alpha: Slope of satellite occupancy
    '''
    from scipy.special import erf
    if sigma == 0.:
        Nc = np.heaviside(M-Mmin, 1.)
    else:
        Nc = 0.5*(1.+erf(np.log10(M/Mmin)/sigma))
    Ns = (np.heaviside(M-M0, 1.)*(M-M0)/M1)**alpha
    return Nc, Ns


def _HOD_Zhai(M:np.ndarray, Mmin=10**13.68, sigma=0.82, Msat=10**14.87, alpha=0.41, Mcut=10**12.32) -> tuple:
    '''
    HOD model from Zhai et al. (2017; https://arxiv.org/abs/1607.05383)\n
    '''
    from scipy.special import erf
    if sigma == 0.:
        Nc = np.heaviside(M-Mmin, 1.)
    else:
        Nc = 0.5*(1.+erf(np.log10(M/Mmin)/sigma))
    Ns = ((M/Msat)**alpha)*np.exp(-Mcut/M) # Paper has a Nc(M) multiplication, but I think the central condition covers this
    return Nc, Ns


def HOD_variance(p:np.ndarray, lam:np.ndarray, central_condition=False) -> tuple:
    '''
    Expected variance (and covariance) in the numbers of central and satellite galaxies\n
    Assumes Bernoulli statistics for centrals and Poisson statistics for satellites\n
    The central condition modifies an 'underlying' Poisson distribution in a calculable way\n
    Args:
        p: Probability of halo hosting a central galaxy
        lam: Mean number of satellite galaxies (iff central_condition=False)
        central_condition: Does a central need to be present in order for there to be a satellite?
    '''
    vcc = p*(1.-p) # Bernoulli
    if central_condition:
        vss = p*lam*(1.+lam*(1.-p)) # Modified Poisson
        vcs = lam*(1.-p)            # Induced covariance
    else:
        vss = lam # Poisson
        vcs = 0.  # No covariance
    return vcc, vss, vcs

### ###