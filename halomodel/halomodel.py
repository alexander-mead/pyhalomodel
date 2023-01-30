# Standard imports
import numpy as np
import scipy.integrate as integrate
import warnings

# Project imports
import utility as util
import cosmology

# To-do list
# TODO: Incorporate configuration-space profiles (high priority)
# TODO: Dedicated test suite against HMcode/HMx for different cosmologies/halo models/redshifts
# TODO: Add more checks for Dv, dc value compatability with mass functions
# TODO: Add covariance between profile amplitudes (low priority; hard and annoying)
# TODO: Ms, ks -> M, k maybe?

# Parameters
dc_rel_tol = 1e-3    # Relative tolerance for checking 'closeness' of delta_c
Dv_rel_tol = 1e-3    # Relative tolerance for checking 'closeness' of Delta_v
Tinker_z_dep = False # Use redshift-dependent parameters for Tinker formula (only calibrated for M200)
Tinker_PBS = False   # Get the bias from the peak-background split, rather than using the calibrated formula

# Halo-model integration scheme for integration in nu
# NOTE: Cannot use Romberg because integration is done in nu, not M, and this will generally be uneven
# TODO: In HMx I use trapezoid because of fast osciallations, maybe should also use that here?
halo_integration = integrate.trapezoid
#halo_integration = integrate.simps

# W(k) integration scheme integration in r
# TODO: Add FFTlog
#win_integration = integrate.trapezoid
#win_integration = integrate.simps
win_integration = integrate.romb # Needs 2^m+1 (integer m) evenly-spaced samples in R
nr = 1+2**7                      # Number of points in r

# Mass function
eps_deriv_mf = 1e-3 # R -> dR for numerical sigma derivative

# beta_NL
do_I11 = True     # Low M1, M2 portion of the integral
do_I12_I21 = True # Low M1 or low M2 portion of the integral

### Class definition ###

class halo_model():
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
            print('Scale factor: %1.3f'%(self.a))
            print('Redshift: %1.3f'%(self.z))
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
                print('Sheth & Tormen (1999) mass function')
                print('p: %1.3f; q: %1.3f; A: %1.4f'%(p, q, self.A_ST))
            if name == 'Sheth, Mo & Tormen (2001)':
                self.a_SMT = 0.707
                self.b_SMT = 0.5
                self.c_SMT = 0.6
                if verbose:
                    print('a: %1.3f; b: %1.3f; c: %1.3f'%(self.a_SMT, self.b_SMT, self.c_SMT))
        elif name == 'Despali et al. (2016)':
            # Despali et al. (2016; https://arxiv.org/abs/1507.05627)
            # Note that this mass function is not correctly normalised
            # Virial halo definition
            # TODO: Check for inconsistencies with Dv, dc
            self.p_ST = 0.2536
            self.q_ST = 0.7689
            self.A_ST = 0.3295*np.sqrt(2.*self.q_ST/np.pi) # Careful with factors of sqrt(2q/pi) in normalisation
            if verbose:
                print('Despali et al. (2016) mass function')
                print('p: %1.3f; q: %1.3f; A: %1.4f'%(self.p, self.q, self.A_ST))
        elif name == 'Tinker et al. (2010)':
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
                print('Tinker et al. (2010) mass function')
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


    def _mass_function_nu(self, nu:float) -> float:
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


    def _linear_bias_nu(self, nu:float) -> float:
        '''
        Halo linear bias b(nu) with nu=delta_c/sigma(M)
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


    def mass_function(self, Ms:np.ndarray, sigmas=None, sigma=None, Pk_lin=None) -> np.ndarray:
        '''
        Calculates n(M), the halo mass function as a function of halo mass
        n(M) is the comoving number-density of haloes per halo mass\n
        Args:
            Ms: Array of halo masses [Msun/h]
        '''
        F = self.multiplicity_function(Ms, sigmas, sigma, Pk_lin)
        return F*self.rhom/Ms**2


    def multiplicity_function(self, Ms:np.ndarray, sigmas=None, sigma=None, Pk_lin=None) -> np.ndarray:
        '''
        Calculates M^2 n(M) / rhobar, the so-called halo multiplicity function
        Note that this is dimensionless
        TODO: Add calculation of dnu_dlnm for sigmas\n
        Args:
            Ms: Array of halo masses [Msun/h]
        '''
        nus = self._peak_height(Ms, sigmas, sigma, Pk_lin)
        Rs = cosmology.Lagrangian_radius(Ms, self.Om_m)
        if Pk_lin is not None:
            dlnsigma2_dlnR = cosmology.dlnsigma2_dlnR(Rs, Pk_lin)
        elif sigma is not None:
            eps = eps_deriv_mf; dRs = Rs*eps # Uses numerical derivative
            dlnsigma2_dlnR = 2.*util.log_derivative(sigma, Rs, dRs)
        else:
            raise ValueError('Error, this currently only works with either P(k) or sigma(R) functions')
        dnu_dlnm = -(nus/6.)*dlnsigma2_dlnR
        return self._mass_function_nu(nus)*dnu_dlnm


    def linear_bias(self, Ms:np.ndarray, sigmas=None, sigma=None, Pk_lin=None) -> np.ndarray:
        '''
        Calculates the linear halo bias as a function of halo mass\n
        Args:
            Ms: Array of halo masses [Msun/h]
        '''
        nus = self._peak_height(Ms, sigmas, sigma, Pk_lin)
        return self._linear_bias_nu(nus)


    def Lagrangian_radius(self, M:float) -> float:
        '''
        Radius [Mpc/h] of a sphere containing mass M in a homogeneous universe\n
        Args:
            M: Halo mass [Msun/h]
        '''
        return cosmology.Lagrangian_radius(M, self.Om_m)


    def average(self, Ms:np.ndarray, fs:np.ndarray, sigmas=None, sigma=None, Pk_lin=None) -> np.ndarray:
        '''
        Calculate the mean of some f(M) over halo mass <f>: int f(M)n(M)dM where n(M) = dn/dM in some notations
        Note that the units of n(M) are [(Msun/h)^{-1} (Mpc/h)^{-3}] so the units of the result are [F (Mpc/h)^{-3}]
        Common use cases:
            <M/rho> = 1 over all halo mass (equivalent to int g(nu)dnu = 1)
            <M^2/rho> = M_NL non-linear halo mass that maximall contributes to one-halo term (not M*)
            <b(M)M/rho> = 1 over all halo mass (equivalent to int g(nu)b(nu)dnu = 1)
            <N(M)> with N the number of galaxies in each halo of mass M; gives mean number density of galaxies
            <b(M)N(M)>/<N(M)> with N the number of galaxies in each halo of mass M; gives mean bias of galaxies\n
        Args:
            Ms: Array of halo masses [Msun/h]
            fs(Ms): Array of function to calculate mean density of (same length as Ms)
            sigmas(M): Optional array of previously calculated nu values corresponding to M
            sigma(R): Optional function to get sigma(R) at z of interest
            Pk_lin(k): Optional function to get linear power at z of interest
        '''
        nus = self._peak_height(Ms, sigmas, sigma, Pk_lin)
        integrand = (fs/Ms)*self._mass_function_nu(nus)
        return halo_integration(integrand, nus)*self.rhom


    def power_spectrum(self, ks:np.ndarray, Ms:np.ndarray, profs:list, Pk_lin, beta=None, sigmas=None, 
        sigma=None, include_shotnoise=False, correct_discrete=True, k_trunc=None, verbose=False) -> tuple:
        '''
        Computes power spectra given that halo model. Returns the two-halo, one-halo and sum terms.
        TODO: Remove Pk_lin dependence?\n
        Inputs
            ks: Array of wavenumbers [h/Mpc]
            Ms: Array of halo masses [Msun/h]
            profiles: List of halo profiles from haloprof class
            Pk_lin(k): Function to evaluate the linear power spectrum [(Mpc/h)^3]
            beta(M1, M2, k): Optional array of beta_NL values at points Ms, Ms, ks
            sigmas(Ms): Optional pre-computed array of linear sigma(M) values corresponding to Ms
            sigma(R): Optional function to evaluate the linear sigma(R)
            shot: Should shot noise contribution be included within discrete spectra?
            discrete: Properly treat discrete tracers with <N(N-1)> rather than <N^2>?
            k_trunc: None or wavenumber [h/Mpc] at which to truncate the one-halo term at large scales
            verbose: verbosity
        '''
        from time import time
        t1 = time() # Initial time

        # Checks
        if type(profs) != list: raise TypeError('N must be list')
        nf = len(profs) # Number of profiles
        for prof in profs:
            if (ks != prof.k).all(): raise ValueError('k arrays must all be identical to those in profiles')
            if (Ms != prof.M).all(): raise ValueError('Mass arrays must be identical to those in profiles')

        # Create arrays of R (Lagrangian radius) and nu values that correspond to the halo mass
        nus = self._peak_height(Ms, sigmas, sigma, Pk_lin)

        # Calculate the missing halo-bias from the low-mass part of the integral
        A = 1.-integrate.quad(lambda nu: self._mass_function_nu(nu)*self._linear_bias_nu(nu), nus[0], np.inf)[0]
        if verbose: print('Missing halo-bias-mass from the low-mass end of the two-halo integrand:', A)
        if A < 0.:  warnings.warn('Warning: Mass function/bias correction is negative!', RuntimeWarning)

        # Shot noise calculations
        PSNs = []
        for p in profs:
            if p.discrete:
                PSN = self._Pk_1h(Ms, nus, p.N/p.norm**2)
            else:
                PSN = 0.
            PSNs.append(PSN)

        # Fill arrays for results
        nk = len(ks)
        Pk_2h_array = np.zeros((nf, nf, nk))
        Pk_1h_array = np.zeros((nf, nf, nk))
        Pk_hm_array = np.zeros((nf, nf, nk))

        # Loop over halo profiles
        for u, pu in enumerate(profs): 
            for v, pv in enumerate(profs):
                if u <= v:
                    for ik, k in enumerate(ks): # Loop over wavenumbers
                        if beta is None: # Two-halo term, treat non-linear halo bias carefully
                            Pk_2h_array[u, v, ik] = self._Pk_2h(Pk_lin, k, Ms, nus, 
                                                        pu.Wk[:, ik], pv.Wk[:, ik], # TODO: Replace with Uk/norm
                                                        pu.mass, pv.mass, A) # TODO: Remove pu.norm from A*pu.norm
                        else:
                            Pk_2h_array[u, v, ik] = self._Pk_2h(Pk_lin, k, Ms, nus, 
                                                        pu.Wk[:, ik], pv.Wk[:, ik], # TODO: Replace with Uk/norm
                                                        pu.mass, pv.mass, A, beta[:, :, ik])
                        if u == v: # and ((discrete and pu.discrete) or (pu.var is not None)): # One-halo term, treat discrete auto case carefully
                            if correct_discrete and pu.discrete: # Treat discrete tracers
                                Wfac = pu.N*(pu.N-1.) # <N(N-1)> for discrete tracers
                            else:
                                Wfac = pu.N**2 # <N^2> for others
                            if pu.var is not None: Wfac += pu.var # Add variance
                            Wprod = Wfac*(pu.Uk[:, ik]/pu.norm)**2 # Multiply by factors of normalisataion and profile
                        else:
                            Wprod = pu.Wk[:, ik]*pv.Wk[:, ik] # TODO: Replace with Uk/norm
                        Pk_1h_array[u, v, ik] = self._Pk_1h(Ms, nus, Wprod)
                    # Shot noise corrections
                    # If '(not discrete) and shot' or 'discrete and (not shot)' no need to do anything as shot noise already correct
                    if (u == v) and pu.discrete:
                        if correct_discrete and include_shotnoise:
                            Pk_1h_array[u, v, :] += PSNs[u] # Need to add shot noise
                        elif (not correct_discrete) and (not include_shotnoise):
                            warnings.warn('Warning: Subtracting shot noise while not treating discreteness properly is dangerous', RuntimeWarning)
                            Pk_1h_array[u, v, :] -= PSNs[u] # Need to subtract shot noise
                    if k_trunc is not None: Pk_1h_array[u, v, :] *= 1.-np.exp(-(ks/k_trunc)**2) # Suppress one-halo term
                    Pk_hm_array[u, v, :] = Pk_2h_array[u, v, :]+Pk_1h_array[u, v, :] # Sum for total
                else:
                    # No need to do these calculations twice
                    Pk_2h_array[u, v, :] = Pk_2h_array[v, u, :]
                    Pk_1h_array[u, v, :] = Pk_1h_array[v, u, :]
                    Pk_hm_array[u, v, :] = Pk_hm_array[v, u, :]

        # Finish
        t2 = time() # Final time
        if verbose: print('Halomodel calculation time [s]:', t2-t1, '\n')
        return Pk_2h_array, Pk_1h_array, Pk_hm_array


    def _Pk_2h(self, Pk_lin, k:float, Ms:np.ndarray, nus:np.ndarray, 
        Wu:float, Wv:float, mass_u:bool, mass_v:bool, A:float, beta=None) -> float:
        '''
        Two-halo term at a specific wavenumber
        '''
        if beta is None:
            I_NL = 0.
        else:
            I_NL = self._I_beta(beta, Ms, nus, Wu, Wv, mass_u, mass_v, A)
        Iu = self._I_2h(Ms, nus, Wu, mass_u, A)
        Iv = self._I_2h(Ms, nus, Wv, mass_v, A)
        return Pk_lin(k)*(Iu*Iv+I_NL)


    def _Pk_1h(self, Ms:np.ndarray, nus:np.ndarray, WuWv:float) -> float:
        '''
        One-halo term at a specific wavenumber
        '''
        integrand = WuWv*self._mass_function_nu(nus)/Ms
        P_1h = halo_integration(integrand, nus)
        P_1h = P_1h*self.rhom
        return P_1h


    def _I_2h(self, Ms:np.ndarray, nus:np.ndarray, W:float, mass:bool, A:float) -> float:
        '''
        Evaluate the integral that appears in the two-halo term
        '''
        integrand = W*self._linear_bias_nu(nus)*self._mass_function_nu(nus)/Ms
        I_2h = halo_integration(integrand, nus)
        if mass:
            I_2h += A*W[0]/Ms[0]
        I_2h = I_2h*self.rhom
        return I_2h


    def _I_beta(self, beta:np.ndarray, Ms:np.ndarray, nus:np.ndarray, Wu:float, Wv:float, 
        massu:bool, massv:bool, A:float) -> float:
        '''
        Evaluates the beta_NL double integral
        TODO: Loops probably horribly inefficient here
        '''
        from numpy import trapz
        integrand = np.zeros((len(nus), len(nus)))
        for iM1, nu1 in enumerate(nus):
            for iM2, nu2 in enumerate(nus):
                if iM2 >= iM1:
                    M1 = Ms[iM1]
                    M2 = Ms[iM2]
                    W1 = Wu[iM1]
                    W2 = Wv[iM2]
                    g1 = self._mass_function_nu(nu1)
                    g2 = self._mass_function_nu(nu2)
                    b1 = self._linear_bias_nu(nu1)
                    b2 = self._linear_bias_nu(nu2)
                    integrand[iM1, iM2] = beta[iM1, iM2]*W1*W2*g1*g2*b1*b2/(M1*M2)
                else:
                    integrand[iM1, iM2] = integrand[iM2, iM1]
        integral = util.trapz2d(integrand, nus, nus)
        if do_I11 and massu and massv:
            integral += beta[0, 0]*(A**2)*Wu[0]*Wv[0]/Ms[0]**2
        if do_I12_I21 and massu:
            integrand = np.zeros(len(nus))
            for iM, nu in enumerate(nus):
                M = Ms[iM]
                W = Wv[iM]
                g = self._mass_function_nu(nu)
                b = self._linear_bias_nu(nu)
                integrand[iM] = beta[0, iM]*W*g*b/M
            integral += (A*Wu[0]/Ms[0])*trapz(integrand, nus)
        if do_I12_I21 and massv:
            for iM, nu in enumerate(nus):
                M = Ms[iM]
                W = Wu[iM]
                g = self._mass_function_nu(nu)
                b = self._linear_bias_nu(nu)
                integrand[iM] = beta[iM, 0]*W*g*b/M
            integral += (A*Wv[0]/Ms[0])*trapz(integrand, nus)
        return integral*self.rhom**2


    def cross_spectrum_at_halo_mass(self, ks:np.ndarray, Ms:np.ndarray, profs:list, Mh:float,
        Pk_lin, beta=None, sigmas=None, sigma=None, verbose=True) -> tuple:
        '''
        Computes the power spectrum of a single tracer crossed with haloes of a specific mass
        TODO: Remove Pk_lin dependence?\n
        Args:
            ks: Array of wavenumbers [h/Mpc]
            Ms: Array of halo masses [Msun/h]
            profs: A list of halo profiles (haloprof class)
            Mh: Halo mass at which to evaluatae cross spectra [Msun/h]
            Pk_lin(k): Function to evaluate the linear power spectrum [(Mpc/h)^3]
            beta(Ms, ks): Optional array of beta_NL values at points Ms, ks
            sigmas(Ms): Optional pre-computed array of linear sigma(M) values corresponding to Ms
            sigma(R): Optional function to evaluate the linear sigma(R)
            verbose: verbosity
        '''
        from time import time
        from scipy.interpolate import interp1d
        t1 = time() # Initial time

        # Checks
        if type(profs) != list: raise TypeError('N must be list of length 2')
        nf = len(profs) # Number of profiles
        for prof in profs:
            if (ks != prof.k).all(): raise ValueError('k arrays must all be identical to those in profiles')
            if (Ms != prof.M).all(): raise ValueError('Mass arrays must be identical to those in profiles')

        # Create arrays of R (Lagrangian radius) and nu values that correspond to the halo mass
        nus = self._peak_height(Ms, sigmas, sigma, Pk_lin)

        # Calculate the missing halo-bias from the low-mass part of the integral
        integrand = self._mass_function_nu(nus)*self._linear_bias_nu(nus)
        A = 1.-halo_integration(integrand, nus)
        if verbose:
            print('Missing halo-bias-mass from two-halo integrand:', A, '\n')

        # Calculate nu(Mh) and W(Mh, k) by interpolating the input arrays
        # NOTE: Wh is not the halo profile, but the profile of the other thing (u) evaluated at the halo mass!
        nu_M_interp = interp1d(np.log(Ms), nus, kind='cubic')
        nuh = nu_M_interp(np.log(Mh))
        Whs = []
        for prof in profs:
            Wh = np.empty_like(prof.Wk[0, :])
            for ik, _ in enumerate(ks):
                WM_interp = interp1d(np.log(Ms), prof.Wk[:, ik], kind='cubic')
                Wh[ik] = WM_interp(np.log(Mh))
            Whs.append(Wh)

        # Combine everything and return
        nk = len(ks)
        Pk_2h_array = np.zeros((nf, nk))
        Pk_1h_array = np.zeros((nf, nk))
        Pk_hm_array = np.zeros((nf, nk))
        for u, prof in enumerate(profs):
            for ik, k in enumerate(ks):
                if beta is None:
                    Pk_2h_array[u, ik] = self._Pk_2h_hu(Pk_lin, k, Ms, nuh, nus, prof.Wk[:, ik], prof.mass, A)
                else:
                    Pk_2h_array[u, ik] = self._Pk_2h_hu(Pk_lin, k, Ms, nuh, nus, prof.Wk[:, ik], prof.mass, A, beta[:, ik])
                Pk_1h_array[ik] = Whs[u][ik] # Simply the halo profile at M=Mh here
                Pk_hm_array[ik] = Pk_2h_array[ik]+Pk_1h_array[ik]
        t2 = time() # Final time

        if verbose:  
            print('Halomodel calculation time [s]:', t2-t1, '\n')

        return Pk_2h_array, Pk_1h_array, Pk_hm_array


    def _Pk_2h_hu(self, Pk_lin, k:float, Ms:np.ndarray, nuh:float, 
        nus:np.ndarray, Wk:float, mass:bool, A:float, beta=None) -> float:
        '''
        Two-halo term for halo-u at a specific wavenumber
        '''
        if beta is None:
            I_NL = 0.
        else:
            I_NL = self._I_beta_hu(beta, Ms, nuh, nus, Wk, mass, A)
        Ih = self._linear_bias_nu(nuh) # Simply the linear bias
        Iu = self._I_2h(Ms, nus, Wk, mass, A) # Same as for the standard two-halo term
        return Pk_lin(k)*(Ih*Iu+I_NL)


    def _I_beta_hu(self, beta:np.ndarray, Ms:np.ndarray, nuh:float, nus:np.ndarray, 
        Wk:np.ndarray, mass:bool, A:float) -> float:
        '''
        Evaluates the beta_NL integral for halo-u
        TODO: Loop probably very inefficent here
        '''
        from numpy import trapz
        bh = self._linear_bias_nu(nuh)
        integrand = np.zeros(len(nus))
        for iM, nu in enumerate(nus):
            M = Ms[iM]
            W = Wk[iM]
            g = self._mass_function_nu(nu)
            b = self._linear_bias_nu(nu)
            integrand[iM] = beta[iM]*W*g*b/M
        integral = trapz(integrand, nus)
        if mass:
            integral += A*beta[0]*Wk[0]/Ms[0]
        return bh*integral*self.rhom


    def _peak_height(self, Ms:np.ndarray, sigmas=None, sigma=None, Pk_lin=None) -> np.ndarray:
        '''
        Calculate peak-height (nu) values from array of halo masses
        '''
        # Create arrays of R (Lagrangian) and nu values that correspond to the halo mass
        Rs = cosmology.Lagrangian_radius(Ms, self.Om_m)

        # Convert R values to nu via sigma(R)
        if sigmas is not None:
            nus = self.dc/sigmas # Use the provided sigma(R) values or...
        elif sigma is not None:
            nus = self.dc/sigma(Rs) # ...otherwise evaluate the provided sigma(R) function or...
        elif Pk_lin is not None:
            nus = self.dc/cosmology.sigmaR(Rs, Pk_lin, integration_type='quad') ### ...otherwise integrate
        else:
            raise ValueError('Error, you need to specify (at least) one of Pk_lin, sigma or sigmas') 
        return nus


    def virial_radius(self, M:float) -> float:
        '''
        Halo virial radius based on the halo mass and overdensity condition
        TODO: Change name virial_radius -> radius?\n
        Args:
            M: Halo mass [Msun/h]
        '''
        return cosmology.Lagrangian_radius(M, self.Om_m)/np.cbrt(self.Dv)

### ###

### Beta_NL ###

def interpolate_beta_NL(ks:np.ndarray, Ms:np.ndarray, Ms_small:np.ndarray, beta_NL_small:np.ndarray, 
    scheme='RegularGridInterp', **kwargs) -> np.ndarray:
    '''
    Wrapper for various beta_NL interpolation schemes to go from coarse grid of halo masses
    to a finer grid of halo masses.
    '''
    if scheme == 'interp2d':
        beta_NL = _interpolate_beta_NL(ks, Ms, Ms_small, beta_NL_small, **kwargs)
    elif scheme == 'RegularGridInterp':
        beta_NL = _RegularGridInterp_beta_NL(Ms_small, ks, beta_NL_small, Ms, ks, **kwargs)
    elif scheme == 'RegularGridInterp_log':
        beta_NL = _RegularGridInterp_beta_NL_log(Ms_small, ks, beta_NL_small, Ms, ks, **kwargs)
    else:
        raise ValueError('beta_NL interpolation method not recognised')
    return beta_NL


def _interpolate_beta_NL(ks:np.ndarray, Ms:np.ndarray, Ms_small:np.ndarray, beta_NL_small:np.ndarray, 
    fill_value=0.) -> np.ndarray:
    '''
    Interpolate beta_NL from a small grid to a large grid for halo-model calculations
    TODO: Remove inefficient loops
    TODO: Add more interesting extrapolations. Currently just nearest neighbour, 
    which means everything outside the range is set to the same value
    '''
    from scipy.interpolate import interp2d
    beta_NL = np.zeros((len(Ms), len(Ms), len(ks))) # Numpy array for output
    for ik, _ in enumerate(ks):
        beta_NL_interp = interp2d(np.log(Ms_small), np.log(Ms_small), beta_NL_small[:, :, ik],
            kind='linear', fill_value=fill_value)
        for iM1, M1 in enumerate(Ms):
            for iM2, M2 in enumerate(Ms):
                beta_NL[iM1, iM2, ik] = beta_NL_interp(np.log(M1), np.log(M2))
    return beta_NL


def _RegularGridInterp_beta_NL(Ms_in:np.ndarray, ks_in:np.ndarray, beta_NL_in:np.ndarray, 
    Ms_out:np.ndarray, ks_out:np.ndarray, method='linear') -> np.ndarray:
    '''
    Supported methods are 'linear', 'nearest', 'slinear', 'cubic', and 'quintic'
    '''
    from scipy.interpolate import RegularGridInterpolator
    bnl_interp = RegularGridInterpolator([Ms_in, Ms_in, np.log(ks_in)], beta_NL_in,
        method=method, fill_value=None, bounds_error=False)
    bnl_out = np.zeros((Ms_out.size, Ms_out.size, ks_out.size))
    indices = np.vstack(np.meshgrid(np.arange(Ms_out.size), np.arange(Ms_out.size), np.arange(ks_out.size), 
                                    copy=False)).reshape(3, -1).T
    values = np.vstack(np.meshgrid(Ms_out, Ms_out, np.log(ks_out), 
                                   copy=False)).reshape(3, -1).T
    bnl_out[indices[:, 0], indices[:, 1], indices[:, 2]] = bnl_interp(values)
    return bnl_out


def _RegularGridInterp_beta_NL_log(Ms_in:np.ndarray, ks_in:np.ndarray, beta_NL_in:np.ndarray, 
    Ms_out:np.ndarray, ks_out:np.ndarray, method='linear') -> np.ndarray:
    '''
    Supported methods are 'linear', 'nearest', 'slinear', 'cubic', and 'quintic'
    '''
    from scipy.interpolate import RegularGridInterpolator
    bnl_interp = RegularGridInterpolator([np.log10(Ms_in), np.log10(Ms_in), np.log(ks_in)], beta_NL_in,
        method=method, fill_value=None, bounds_error=False)
    bnl_out = np.zeros((Ms_out.size, Ms_out.size, ks_out.size))
    indices = np.vstack(np.meshgrid(np.arange(Ms_out.size), np.arange(Ms_out.size), np.arange(ks_out.size), 
                                    copy=False)).reshape(3, -1).T
    values = np.vstack(np.meshgrid(np.log10(Ms_out), np.log10(Ms_out), np.log(ks_out), 
                                   copy=False)).reshape(3, -1).T
    bnl_out[indices[:, 0], indices[:, 1], indices[:, 2]] = bnl_interp(values)
    return bnl_out

### ###

### Halo profiles ###

class halo_profile():
    '''
    Class for halo profiles
    '''
    def __init__(self, ks:np.ndarray, Ms:np.ndarray, Ns:np.ndarray, Uk:np.ndarray, 
        norm=1., var=None, Prho=None, mass=False, discrete=False, *args):
        '''
        Class initialisation for halo profiles
        TODO: Allow for configuration-space profiles to be specified\n
        Args:
            ks: array of wavenumbers [h/Mpc] going from low to high
            Ms: array of halo masses [Msun/h] going from low to high
            Ns(M): 1D array of halo profile amplitudes at halo masses 'M' (e.g., M for mass; N for galaxies)
            Uk(M, k): 2D array of normalised halo Fourier transform [dimensionless]; should have U(M, k->0) = 1
            norm: float of normalisation (e.g., rhom for mass, ng for galaxies)
            var(M): Var(N(M)) (auto)variance in the profile amplitude at each halo mass (e.g., N for Poisson galaxies)
            Prho(r, rv, *args): 4\pi r^2\rho(r, rv, *args) for density profile
            mass: flag to determine if contributions are expected for M < M[0] (e.g., matter)
            discrete: does the profile correspond to that of a discrete tracer (e.g., galaxies)
            *args: Arguments for Prho
        '''
        # Set internal variables
        self.k = np.copy(ks)
        self.M = np.copy(Ms)
        self.mass = mass
        self.discrete = discrete
        self.norm = norm
        if Prho is None:
            self.N = np.copy(Ns)
            self.Uk = np.copy(Uk)
            if var is None:
                self.var = None
            else:
                self.var = np.copy(var)
            # Calculations; TODO: Do I need Wk?
            self.Wk = (self.Uk.T*self.N).T/self.norm # Transposes necessary to get multiplication correct
        else:
            # Calculate the halo profile Fourier transform
            self.N = np.zeros(len(Ms))
            self.Uk = np.zeros((len(Ms), len(ks)))
            self.Wk = np.zeros((len(Ms), len(ks)))
            for iM, _ in enumerate(Ms):
                rv = 1. # TODO: This
                self.N[iM] = self._halo_window(0., rv, Prho, *args)
                for ik, k in enumerate(ks):
                    self.Wk[iM, ik] = self._halo_window(k, rv, Prho, *args)
                    self.Uk[iM, ik]= self.Wk[iM, ik]/self.N[iM] # Always normalised halo profile
            if Ns is not None:
                self.N = np.copy(Ns)
                self.Wk = self.Uk*self.N # Renormalise halo profile
            self.Wk /= norm # TODO: Check /= operation, want A = A/b


    def _halo_window(k:float, rv:float, Prho, *args) -> float:
        '''
        Compute the halo window function given a 'density' profile Prho(r) = 4*pi*r^2*rho(r)
        TODO: Use integration scheme for continuous function 
        TODO: This should almost certainly be done with a dedicated integration routine, FFTlog?\n
        Args:
            k: Fourier wavenumber [h/Mpc]
            rv: Halo virial radius [Mpc/h]
            Prho(r, rv, *args): Function Prho = 4pi*r^2*rho values at different radii
        '''
        from scipy.special import spherical_jn

        # Spacing between points for Romberg integration
        rs = np.linspace(0., rv, nr)
        dr = rs[1]-rs[0]

        # Calculate profile mean
        integrand = spherical_jn(0, k*rs)*Prho(rs, rv, *args)
        if win_integration == integrate.romb:
            Wk = win_integration(integrand, dr)
        elif win_integration in [integrate.trapezoid, integrate.simps]:
            Wk = win_integration(integrand, rs)
        else:
            raise ValueError('Halo window function integration method not recognised')
        return Wk

### Halo profiles in configuration space ###

# def _Prho_isothermal(r, rv, M):
#     '''
#     Isothermal density profile multiplied by 4*pi*r^2
#     '''
#     return M/rv


# def _Prho_NFW(r, rv, M, c):
#     '''
#     NFW density profile multiplied by 4*pi*r^2
#     '''
#     rs = rv/c
#     return M*r/(_NFW_factor(c)*(1.+r/rs)**2*rs**2)


# def Prho_UPP(r, r500, M, z, Om_r, Om_m, Om_w, Om_v, h):
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


# def _rho_from_Prho(Prho, r, *args):
#     '''
#     Converts a Prho profile to a rho profile
#     Take care evaluating this at zero (which will give infinity)
#     '''
#     return Prho(r, *args)/(4.*np.pi*r**2)


# def rho_isothermal(r, rv, M):
#     '''
#     Density profile for an isothermal halo
#     '''
#     return _rho_from_Prho(_Prho_isothermal, r, rv, M)


# def rho_NFW(r, rv, M, c):
#     '''
#     Density profile for an NFW halo
#     '''
#     return _rho_from_Prho(_Prho_NFW, r, rv, M, c)

### ###

### Halo profiles in Fourier space ###

def halo_window_function(k:float, rv:float, *args, profile='NFW'):
    '''
    Normalised Fourier tranform for a delta-function profile (unity)\n
    Args:
        k: Fourier wavenumber [h/Mpc]
        rv: Halo virial radius [Mpc/h]
        profile: delta, isothermal, NFW
        *args: Additional arguments for halo profiles (e.g., concentration)
    '''
    if profile == 'delta':
        Wk = 1.
    elif profile == 'isothermal':
        Wk = _win_isothermal(k, rv)
    elif profile == 'NFW':
        Wk = _win_NFW(k, rv, *args)
    else:
        raise ValueError('Halo profile not recognised')
    return Wk


def _win_isothermal(k:float, rv:float) -> float:
    '''
    Normalised Fourier transform for an isothermal profile\n
    Args:
        k: Fourier wavenumber [h/Mpc]
        rv: Halo virial radius [Mpc/h]
    '''
    from scipy.special import sici
    Si, _ = sici(k*rv)
    return Si/(k*rv)


def _win_NFW(k:float, rv:float, c:float) -> float:
    '''
    Normalised Fourier transform for an NFW profile\n
    Args:
        k: Fourier wavenumber [h/Mpc]
        rv: Halo virial radius [Mpc/h]
        c: Halo concentration
    '''
    from scipy.special import sici
    rs = rv/c
    kv = k*rv
    ks = k*rs
    Sisv, Cisv = sici(ks+kv)
    Sis, Cis = sici(ks)
    f1 = np.cos(ks)*(Cisv-Cis)
    f2 = np.sin(ks)*(Sisv-Sis)
    f3 = np.sin(kv)/(ks+kv)
    f4 = _NFW_factor(c)
    return (f1+f2-f3)/f4


def _NFW_factor(c:float) -> float:
    '''
    Factor from normalisation that always appears in NFW equations\n
    Args:
        c: Halo concentration
    '''
    return np.log(1.+c)-c/(1.+c)


def matter_profile(ks:np.ndarray, Ms:np.ndarray, rvs:np.ndarray, cs:np.ndarray, Om_m:float) -> halo_profile:
    '''
    Pre-configured matter NFW profile\n
    Args:
        ks: Array of wavenumbers [h/Mpc]
        Ms: Array of halo masses [Msun/h]
        rvs: Array of halo virial radii [Mpc/h]
        cs: Array of halo concentrations
        Om_m: Cosmological matter density
    '''
    rhom = cosmology.comoving_matter_density(Om_m)
    Uk = np.zeros((len(Ms), len(ks)))
    for iM, (rv, c) in enumerate(zip(rvs, cs)):
        Uk[iM, :] = _win_NFW(ks, rv, c)
    return halo_profile(ks, Ms, Ms, Uk, rhom, mass=True, discrete=False)


def galaxy_profile(ks:np.ndarray, Ms:np.ndarray, rvs:np.ndarray, cs:np.ndarray, rhog:float, 
    HOD_method='Zheng et al. (2005)') -> halo_profile:
    '''
    Pre-configured galaxy profile\n
    # TODO: Need rhog here, but to get this requires integrating Ns, which would require hmod as argument
    Args:
        ks: Array of wavenumbers [h/Mpc]
        Ms: Array of halo masses [Msun/h]
        rvs: Array of halo virial radii [Mpc/h]
        cs: Array of halo concentrations
        rhog: Galaxy number density []
        HOD_method: String for HOD choice
    '''
    N_cen, N_sat = HOD_mean(Ms, method=HOD_method)
    N_gal = N_cen+N_sat
    nM, nk = len(Ms), len(ks)
    #Uk_cen = np.zeros((nM, nk)); Uk_sat = np.zeros((nM, nk))
    Uk_gal = np.zeros((nM, nk))
    for iM, (rv, c) in enumerate(zip(rvs, cs)):
        #Uk_cen[iM, :] = halo_window_function(ks, rv, profile='delta')
        #Uk_sat[iM, :] = halo_window_function(ks, rv, c, profile='NFW')
        Uk_gal[iM, :] = halo_window_function(ks, rv, profile='isothermal')
    #profile_cen = halo_profile(ks, Ms, N_cen, Uk_cen, rho_gal, var=None, Prho=None, mass=True, discrete=False)
    #profile_sat = halo_profile(ks, Ms, N_sat, Uk_sat, rho_gal, var=None, Prho=None, mass=True, discrete=False)
    profile = halo_profile(ks, Ms, N_gal, Uk_gal, rhog, var=None, Prho=None, mass=True, discrete=False)
    return profile

### ###

### Halo concentration

def concentration(M:float, z:float, method='Duffy et al. (2008)', halo_definition='M200') -> float:
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

def _concentration_Duffy(M:float, z:float, halo_definition='M200') -> float:
    '''
    Duffy et al (2008; 0804.2486) c(M) relation for WMAP5, See Table 1
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

def _HOD_simple(M:float, Mmin=1e12, Msat=1e13, alpha=1.) -> tuple:
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


def _HOD_Zehavi(M:float, Mmin=1e12, M1=1e13, alpha=1.) -> tuple:
    '''
    HOD model from Zehavi et al. (2004; https://arxiv.org/abs/astro-ph/0703457)
    Same as Zheng model in the limit that sigma=0 and M0=0
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


def _HOD_Zheng(M:float, Mmin=1e12, sigma=0.15, M0=1e12, M1=1e13, alpha=1.) -> tuple:
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


def _HOD_Zhai(M:float, Mmin=10**13.68, sigma=0.82, Msat=10**14.87, alpha=0.41, Mcut=10**12.32) -> tuple:
    '''
    HOD model from Zhai et al. (2017; https://arxiv.org/abs/1607.05383)
    '''
    from scipy.special import erf
    if sigma == 0.:
        Nc = np.heaviside(M-Mmin, 1.)
    else:
        Nc = 0.5*(1.+erf(np.log10(M/Mmin)/sigma))
    Ns = ((M/Msat)**alpha)*np.exp(-Mcut/M) # Paper has a Nc(M) multiplication, but I think the central condition covers this
    return Nc, Ns


def HOD_variance(p:float, lam:float, central_condition=False) -> tuple:
    '''
    Expected variance (and covariance) in the numbers of central and satellite galaxies
    Assumes Bernoulli statistics for centrals and Poisson statistics for satellites
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