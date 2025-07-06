# Author: Parth Nayak (LMU Munich)
# Synthesis of Transmission from Hydrodynamic simulations (SynTH)

# This module provides utilities for synthesizing Lya transmission spectra from hydrodynamic simulations 

import os
import numpy as np
import h5py
import yaml
# from numpy.fft import rfft,rfftfreq
from scipy.special import erf, voigt_profile
from scipy.integrate import simpson
import synth.atomic_rates as AR
from synth.constants import *
from scipy.interpolate import interp1d, RectBivariateSpline

cwd = os.path.dirname(__file__)


def cosmology(hfilepath):
    """
    Read the cosmological parameters and domain properties from an input skewers HDF5 file.

    Args:
        hfilepath (pathlike): path to the input skewers HDF5 file 

    Returns:
        Cosmology (dict): dictionary containing cosmological parameters with keys: 'h', 'Omega_baryons', 'Omega_matter', 'Omega_Lambda', 'redshift'
        Domain (dict): dictionary containing domain properties with keys: 'N_cells', 'side_length', 'side_length_Mpc'
    """
    with h5py.File(hfilepath, 'r') as f:
        h, Omega_m, Omega_L, Omega_b, z = f['cosmo_prop'][:]
        domain = f['domain'][:]
        Cosmology = {
            'h': h,
            'Omega_baryons': Omega_b,
            'Omega_matter': Omega_m,
            'Omega_Lambda': Omega_L,
            'redshift': z
        }
        Domain = {
            'N_cells': domain[0],  # in number of cells
            'side_length': domain[1],   # in Mpc/h
            'side_length_Mpc': domain[1]/h, 
        }
    return Cosmology, Domain



class Hubble:
    """    
    Class to compute the Hubble flow velocity for any given skewer (or pixel) for a simulation box.
    This class reads the cosmological parameters from a file and computes the Hubble flow velocity based on the redshift and the cosmological parameters.
    Args:
        hfilepath (pathlike): path to either the input or the extracted skewers HDF5 file
    """
    def __init__(self, hfilepath):
        with h5py.File(hfilepath, 'r') as hfile:
            self.cosmo_prop = hfile['cosmo_prop'][:]
            self.domain     = hfile['domain'][:]
        self.h, self.Omega_m, self.Omega_L, self.Omega_b, self.z = self.cosmo_prop
        self.H0     = 100.* self.h / 3.086E19       # in CGS (s^-1)
        self.Hz 	= self.H0 * np.sqrt(self.Omega_m*(1 + self.z)**3 + self.Omega_L)
            
    def v_hubble(self, p):
        """
        Compute the Hubble flow velocity for a given skewer (or pixel).
        Args:
            p (float [ndarray]): index of the pixel(s) of a given skewer. Integers mark the pixel centers.
        Returns:
            float [ndarray]: Hubble flow velocity in CGS (cm/s)
        """         	
        Delta_l		= self.domain[1] / (self.h * self.domain[0])  #comoving length of each cell in Mpc
        Delta_l    	= Delta_l * 3.086E24    					  #in CGS (cm)  				
        return self.Hz * Delta_l * (p+0) /(1 + self.z)			  


class LineProfile:
    """
    This class provides methods to compute the spectroscopic line profiles: Lorentz, Doppler, and Voigt.
    Args:
        v0 (float [ndarray]):        line center velocity
        sigma (float [ndarray]):     Doppler width (standard deviation)
        gamma (float):               Lorentzian damping width
        dv_cells (float [ndarray]):  velocity width of the given cell(s) in the skewer (in the same units as v0, sigma, and gamma), should be the same size as v0 (using periodic boundary conditions); supply only for the error function approximation of the Doppler profile.
    """
    def __init__(self, v0, sigma, gamma, dv_cells = None):
        self.sigma = sigma
        self.gamma = gamma
        self.v0 = v0
        self.dv_cells = dv_cells
    
    def voigt(self, v):
        """
        Compute the Voigt profile for a given velocity array.
        Args:
            v (float [ndarray]): velocity array (in the same units as v0, sigma, and gamma)
        Returns:
            float [ndarray]: Voigt profile values at the given output pixel velocities
        """
        return voigt_profile(v-self.v0, self.sigma, self.gamma)
    
    def lorentz(self, v):
        """
        Compute the Lorentz profile for a given velocity array.
        Args:
            v (float [ndarray]): velocity array (in the same units as v0, sigma, and gamma)
        Returns:
            float [ndarray]: Lorentz profile values at the given output pixel velocities
        """
        return self.gamma / (self.gamma**2 + (v-self.v0)**2) / np.pi
        
    def doppler(self, v): 
        """
        Compute the Doppler profile for a given velocity array.
        Args:
            v (float [ndarray]): velocity array (in the same units as v0, sigma, and gamma)
        Returns:
            float [ndarray]: Doppler profile values at the given output pixel velocities
        """
        return np.exp(-(v-self.v0)**2/(2.*self.sigma**2)) / (self.sigma * np.sqrt(2.*np.pi))
    
    def gauss(self, nsig): 
        return np.exp(-nsig**2/2.)
    
    def error_function(self, v):
        """
        Compute the error function approximation of the Doppler profile for a given velocity array.
        Args:
            v (float [ndarray]): velocity array (in the same units as v0, sigma, and gamma)
        Returns:
            float [ndarray]: error function approximation of the profile at the given output pixel velocities
        """
        assert self.dv_cells is not None, "dv_cells must be provided for the error function approximation of the Doppler profile."
        y_l  = (v - self.v0 + self.dv_cells/2.) / (self.sigma * np.sqrt(2.))
        y_u  = (v - self.v0 - self.dv_cells/2.) / (self.sigma * np.sqrt(2.))
        prof = 0.5 * (erf(y_l) - erf(y_u))
        return prof

    def create_interpolator_doppler(self,nsig,npix):
        """
        Create an interpolator for the normalized Doppler profile.
        Args:
            nsig (float):       number of standard deviations around the mean as limits for the profile interpolation
            npix (int):         number of pixels in the profile
        Returns:
            function:           interpolator function for the normalized Doppler profile, which takes velocity (much the same way as `doppler`) as an argument and returns the interpolated normalized profile value.
        """
        nsigarr  = np.linspace(-nsig, nsig, npix)
        profile  = self.gauss(nsigarr)
        interp_g = interp1d(nsigarr, profile, fill_value =0., bounds_error = False)
        self.interp_norm_doppler = lambda v: interp_g((v-self.v0)/self.sigma) / (self.sigma * np.sqrt(2.*np.pi))
        return self.interp_norm_doppler
        
    def create_interpolator_voigt(self, nsig, min_sig, max_sig, Npix_nsig, Npix_sig):
        """
        Create an interpolator for the normalized Voigt profile.
        Args:
            nsig (float):       number of standard deviations around the mean as limits for the profile interpolation
            min_sig (float):    minimum value of sigma for the profile interpolation
            max_sig (float):    maximum value of sigma for the profile interpolation
            Npix_nsig (int):    number of pixels in the nsig array
            Npix_sig (int):     number of pixels in the sigma array
        Returns:
            function:           interpolator function for the normalized Voigt profile, which takes velocity (much the same way as `voigt`) as an argument and returns the interpolated normalized profile value.
        """
        nsigarr = np.linspace(-nsig, nsig, Npix_nsig)
        sigarr  = np.linspace(min_sig, max_sig, Npix_sig)
        nsigarr2d, sigarr2d = np.meshgrid(nsigarr, sigarr)
        profile2d = voigt_profile(nsigarr2d*sigarr2d, sigarr2d, self.gamma).T
        #interp_v  = interp2d(nsigarr, sigarr, profile2d, bounds_error = False, fill_value = 0.)
        interp_v  = RectBivariateSpline(nsigarr, sigarr, profile2d, kx = 3, ky = 3) #, bounds_error = False, fill_value = 0.)
        self.interp_norm_voigt = lambda v: interp_v((v-self.v0)/self.sigma, self.sigma, grid = False) #np.array([interp_v(v/s, s) for v,s in zip(dv,sigma)])
        return self.interp_norm_voigt



class IonizationEquilibrium:
    """
    Computations of quantities of various species of Hydrogen and Helium under the assumption of ionization equilibrium
    """
    def __init__(self, Cosmology, X, Y, TREECOOL = None):
        """
        Initialize with the underlying Cosmological parameters, Hydrogen & Heliuem abundances

        Args:
            Cosmology (dict):       dictionary of the cosmological parameters, the required keys are 'h', 'Omega_baryons', 'Omega_matter', 'Omega_Lambda', 'redshift'
            X (float):              mass abundance (fraction) of Hydrogen 
            Y (float):              mass abundance (fraction) of Helium 
            TREECOOL (str, optional):   full path to the TREECOOL UVB table file 
        """
        self.H_0        = 100 * Cosmology['h'] / 3.086E19  # in s^-1
        self.z          = Cosmology['redshift']
        self.Omega_b    = Cosmology['Omega_baryons']
        self.Omega_m    = Cosmology['Omega_matter']
        self.Omega_L    = Cosmology['Omega_Lambda']
        self.Hz 	    = self.H_0 * np.sqrt(self.Omega_m*(1 + self.z)**3 + self.Omega_L)
        self.critical_density = 3. * self.H_0**2 / (8. * np.pi * G)
        self.X          = X
        self.Y          = Y 
        if TREECOOL is None: 
            TREECOOL   = os.path.join(cwd, 'TREECOOL_ONO16_C000_T632.txt')
            # TREECOOL    = 'TREECOOL_ONO16_C000_T632.txt'
        Gamma_table     = np.loadtxt(TREECOOL)
        Gamma_table[:,0] = 10**(Gamma_table[:,0]) - 1
        ind             = np.argmin(np.abs(Gamma_table[:,0] - self.z))
        self.HI_UV      = Gamma_table[ind, 1]
        self.HeI_UV     = Gamma_table[ind, 2]
        self.HeII_UV    = Gamma_table[ind, 3]

    def Alpha_r_HII(self, T):
        """
        Recombination rate of HII

        Args:
            T (float [ndarray]): gas temperature in K

        Returns:
            float [ndarray]: Recombination rate in CGS (cm^3 s^-1)
        """
        return AR.Alpha_r(a = 7.982E-11, b = 0.7480, T0 = 3.148, T1 = 7.036E5).fit(T)
    
    def Alpha_r_HeII(self, T):
        """
        Recombination rate of HeII

        Args:
            T (float [ndarray]): gas temperature in K

        Returns:
            float [ndarray]: Recombination rate in CGS (cm^3 s^-1)
        """
        if type(T) is float or type(T) is np.float_:
            return AR.Alpha_r(a = 3.294E-11, b =  0.6910, T0 = 15.54, T1 = 3.676E7).fit(T)
        elif type(T) is np.ndarray:
            return AR.Alpha_r(a = 3.294E-11, b =  0.6910, T0 = 15.54, T1 = 3.676E7).alternative(T, a1 = 9.356E-10, b1 = 0.7892, T01 = 4.266E-2, T11 = 4.677E6)
        
    def Alpha_r_HeIII(self, T):
        """
        Recombination rate of HeIII

        Args:
            T (float [ndarray]): gas temperature in K

        Returns:
            float [ndarray]: Recombination rate in CGS (cm^3 s^-1)
        """
        return AR.Alpha_r(a = 1.891E-10, b = 0.7524, T0 = 9.370, T1 = 2.774E6).fit(T)
    
    def Alpha_d_HeIII(self, T):
        """
        Dielectric recombination rate of HeIII

        Args:
            T (float [ndarray]): gas temperature in K

        Returns:
            float [ndarray]: Recombination rate in CGS (cm^3 s^-1)
        """
        return AR.Alpha_d(T)
    
    def Gamma_c_HI(self, T):
        """
        Collisional ionization rate of HI

        Args:
            T (float [ndarray]): gas temperature in K

        Returns:
            float [ndarray]: Ionization rate in CGS (cm^3 s^-1)
        """
        return AR.Gamma_coll(m = 0.39, A = 2.91E-8, P = 0., E = 13.6, X = 0.232).fit(T) 
    
    def Gamma_c_HeI(self, T):
        """
        Collisional ionization rate of HeI

        Args:
            T (float [ndarray]): gas temperature in K

        Returns:
            float [ndarray]: Ionization rate in CGS (cm^3 s^-1)
        """
        return AR.Gamma_coll(m = 0.35, A = 1.75E-8, P = 0., E = 24.6, X = 0.180).fit(T) 

    def Gamma_c_HeII(self, T):
        """
        Collisional ionization rate of HeII

        Args:
            T (float [ndarray]): gas temperature in K

        Returns:
            float [ndarray]: Ionization rate in CGS (cm^3 s^-1)
        """
        return AR.Gamma_coll(m = 0.25, A = 2.05E-9, P = 1., E = 54.4, X = 0.265).fit(T)
    


    def eval_fractions_iteratively(self, rho_b, T, n_iter = 2):
        """
        Evaluate mass fractions of all the different neutral & ionized species of H and He iteratively

        Args:
            rho_b (float [ndarray]):    value of local baryon overdensity, i.e., (density / mean_density)
            T (float [ndarray]):        value of local gas temperature (in K)
            n_iter (int):               number of iterations to perform for convergence

        Returns:
            dict: Dictionary of all the mass fractions (floats). The keys are 'x_HI', 'x_HII', 'x_HeI', 'x_HeII', 'x_HeIII'.
        """
        Rho_b   = rho_b * self.Omega_b * self.critical_density * (1+self.z)**3 
        A  	    = Rho_b * self.X / mH
        B       = Rho_b * self.Y / (2. * mH)
        nH      = A
        nHe     = B / 2.

        GcHI    = self.Gamma_c_HI(T)
        GcHeI   = self.Gamma_c_HeI(T)
        GcHeII  = self.Gamma_c_HeII(T)
        
        ArHII   = self.Alpha_r_HII(T)
        ArHeII  = self.Alpha_r_HeII(T)
        ArHeIII = self.Alpha_r_HeIII(T)
        AdHeII  = self.Alpha_d_HeIII(T)
        DHeII   = GcHeII + ArHeII + AdHeII

        if type(T) is float or type(T) is np.float_:
            GgHI   = self.HI_UV  
            GgHeI  = self.HeI_UV 
            GgHeII = self.HeII_UV 
        elif type(T) is np.ndarray:
            GgHI   = np.ones(T.shape) * self.HI_UV  #1.0E-12 
            GgHeI  = np.ones(T.shape) * self.HeI_UV #1.0E-13 
            GgHeII = np.ones(T.shape) * self.HeII_UV #1.0E-15 

        ## First solve for xHeII under the assumption of complete ionization, i.e., x_HI = xHeI = 0.
        a  = B*(ArHeIII+DHeII)/2.
        b  = -(GgHeII + ArHeIII*(A+3*B/2) + DHeII*(A+B))
        c  = ArHeIII * (A + B)
        D  = b**2 - 4.*a*c 
        xHeII  = (-b - np.sqrt(D))/(2.*a)  # this is the physically meaningful solution of the two solutions of the quadratic equation
        xHeIII = 1. - xHeII
        n_e    = A + B - B*xHeII/2.
        xHI    = ArHII * n_e / ((ArHII + GcHI)*n_e + GgHI)
        xHeI   = xHeII * n_e*(ArHeII + AdHeII)/(GcHeI*n_e + GgHeI)
        n_e    = A + B - A*xHI - B*xHeI - B*xHeII/2.
        xHI    = ArHII * n_e / ((ArHII + GcHI)*n_e + GgHI) # -------------> First iteration done.
        xHII   = 1 - xHI

        iter_id = 1
        while iter_id < n_iter:
            iter_id += 1
            
            xHeII  = xHeIII * ArHeIII*n_e / (GcHeII*n_e + GgHeII)
            xHeI   = xHeII * n_e*(ArHeII + AdHeII)/(GcHeI*n_e + GgHeI)
            xHeIII = 1. - xHeI - xHeII
            n_e    = A + B - A*xHI - B*xHeI - B*xHeII/2.
            xHI    = ArHII * n_e / ((ArHII + GcHI)*n_e + GgHI)
            xHII   = 1 - xHI

        self.nH  = nH 
        self.nHe = nHe
        self.fractions = {'x_HI': xHI, 'x_HII': xHII, 'x_HeI': xHeI, 'x_HeII': xHeII, 'x_HeIII': xHeIII}
        return self.fractions
    
    
    def eval_fractions_iteratively_with_convergence(self, rho_b, T, r_tol = 1e-5):
        """
        Evaluate mass fractions of all the different neutral & ionized species of H and He iteratively until converged

        Args:
            rho_b (float [ndarray]):    value of local baryon overdensity, i.e., (density / mean_density)
            T (float [ndarray]):        value of local gas temperature (in K)
            n_iter (int):               number of iterations to perform for convergence

        Returns:
            dict: Dictionary of all the mass fractions (floats). The keys are 'x_HI', 'x_HII', 'x_HeI', 'x_HeII', 'x_HeIII'.
        """
        Rho_b   = rho_b * self.Omega_b * self.critical_density * (1+self.z)**3 
        A  	    = Rho_b * self.X / mH
        B       = Rho_b * self.Y / (2. * mH)
        nH      = A
        nHe     = B / 2.

        GcHI    = self.Gamma_c_HI(T)
        GcHeI   = self.Gamma_c_HeI(T)
        GcHeII  = self.Gamma_c_HeII(T)
        
        ArHII   = self.Alpha_r_HII(T)
        ArHeII  = self.Alpha_r_HeII(T)
        ArHeIII = self.Alpha_r_HeIII(T)
        AdHeII  = self.Alpha_d_HeIII(T)
        DHeII   = GcHeII + ArHeII + AdHeII

        if type(T) is float or type(T) is np.float_:
            GgHI   = self.HI_UV  
            GgHeI  = self.HeI_UV 
            GgHeII = self.HeII_UV 
        elif type(T) is np.ndarray:
            GgHI   = np.ones(T.shape) * self.HI_UV  #1.0E-12 
            GgHeI  = np.ones(T.shape) * self.HeI_UV #1.0E-13 
            GgHeII = np.ones(T.shape) * self.HeII_UV #1.0E-15 

        ## First solve for xHeII under the assumption of complete ionization, i.e., x_HI = xHeI = 0.
        a  = B*(ArHeIII+DHeII)/2.
        b  = -(GgHeII + ArHeIII*(A+3*B/2) + DHeII*(A+B))
        c  = ArHeIII * (A + B)
        D  = b**2 - 4.*a*c 
        self.xHeII  = (-b - np.sqrt(D))/(2.*a)  # this is the physically meaningful solution of the two solutions of the quadratic equation
        self.xHeIII = 1. - self.xHeII
        self.n_e    = A + B - B*self.xHeII/2.
        self.xHI    = ArHII * self.n_e / ((ArHII + GcHI)*self.n_e + GgHI)
        self.xHeI   = self.xHeII * self.n_e*(ArHeII + AdHeII)/(GcHeI*self.n_e + GgHeI)
        self.n_e    = A + B - A*self.xHI - B*self.xHeI - B*self.xHeII/2.
        self.xHI    = ArHII * self.n_e / ((ArHII + GcHI)*self.n_e + GgHI) # -------------> First iteration done.
        self.xHII   = 1 - self.xHI

        iter_id = 1
        continue_iterating = True
        while continue_iterating:
            iter_id += 1
            
            self.xHeII  = self.xHeIII * ArHeIII*self.n_e / (GcHeII*self.n_e + GgHeII)
            self.xHeI   = self.xHeII * self.n_e*(ArHeII + AdHeII)/(GcHeI*self.n_e + GgHeI)
            self.xHeIII = 1. - self.xHeI - self.xHeII
            self.n_e    = A + B - A*self.xHI - B*self.xHeI - B*self.xHeII/2.
            xHI    = ArHII * self.n_e / ((ArHII + GcHI)*self.n_e + GgHI)
            if np.all(np.abs(xHI/self.xHI - 1) < r_tol) & np.all((0<xHI) & (xHI<1)):
                continue_iterating = False
            self.xHI    = xHI
            self.xHII   = 1 - self.xHI

        self.nH  = nH 
        self.nHe = nHe
        self.fractions = {'x_HI': self.xHI, 'x_HII': self.xHII, 'x_HeI': self.xHeI, 'x_HeII': self.xHeII, 'x_HeIII': self.xHeIII}
        return self.fractions


    def eval_fractions(self, rho_b, T):
        """
        Evaluate mass fractions of all the different neutral & ionized species of H and He

        Args:
            rho_b (float [ndarray]):    value of local baryon overdensity, i.e., (density / mean_density)
            T (float [ndarray]):        value of local gas temperature (in K) 

        Returns:
            dict: Dictionary of all the mass fractions (floats). The keys are 'x_HI', 'x_HII', 'x_HeI', 'x_HeII', 'x_HeIII'.
        """
        from scipy.optimize import root
        
        Rho_b   = rho_b * self.Omega_b * self.critical_density * (1+self.z)**3 
        A  	    = Rho_b * self.X / mH
        B       = Rho_b * self.Y / (2. * mH)
        nH      = A
        nHe     = B / 2.

        def eval(A,B,GcHI,GcHeI,GgH1,GgHe1,GgHe2,ArHII,ArHeII,ArHeIII,AdHeII,DHeII):
        # for i in range(len(T_skewer)):
            P00 = GcHI + ArHII;                     P01 = 2.*P00;           P02 = P00
            P0  = -1.*(P00*(A+B) + ArHII*A + GgH1); P1  = -2.* ArHII * A;   P2  = -1.* ArHII * A;   p = ArHII * A * (A+B)

            Q01 = -1.*GcHeI;            Q11 = -2.*GcHeI;        Q02 = ArHeII + AdHeII;      Q22 = Q02;      Q12 = 2.*Q02 + Q01
            Q1  = GcHeI*(A+B) + GgHe1;  Q2  = -1.*Q02*(A+B)

            R01 = GcHeI - ArHeIII;      R02 = -1.*(DHeII + ArHeIII);            R12 = GcHeI -2.*DHeII - 3.*ArHeIII;     R11 = 2.*R01;       R22 = R02
            R0  = ArHeIII * B / 2.;     R1  = -1.*R01*(A+B) + 2.*R0 - GgHe1;    R2  = -1.*R02*(A+B) + R0 + GgHe2;       r   = -1.*R0*(A+B)

            eq_P = lambda x: P00*x[0]*x[0] + P01*x[0]*x[1] + P02*x[0]*x[2] + P0*x[0] + P1*x[1] + P2*x[2] + p

            eq_Q = lambda x: Q01*x[0]*x[1] + Q11*x[1]*x[1] + Q02*x[0]*x[2] + Q22*x[2]*x[2] + Q12*x[1]*x[2] + Q1*x[1] + Q2*x[2]

            eq_R = lambda x: R01*x[0]*x[1] + R02*x[0]*x[2] + R12*x[1]*x[2] + R11*x[1]*x[1] + R22*x[2]*x[2] + R0*x[0] + R1*x[1] + R2*x[2] + r

            cost = lambda x: np.array([eq_P(x), eq_Q(x), eq_R(x)])

            def Jacobian(x):
                J00 = 2.*P00*x[0] + P01*x[1] + P02*x[2] + P0
                J01 = P01*x[0] + P1
                J02 = P02*x[0] + P2

                J10 = Q01*x[1] + Q02*x[2]
                J11 = Q01*x[0] + 2.*Q11*x[1] + Q12*x[2] + Q1
                J12 = Q02*x[0] + 2.*Q22*x[2] + Q12*x[1] + Q2

                J20 = R01*x[1] + R02*x[2] + R0
                J21 = R01*x[0] + R12*x[2] + 2.*R11*x[1] + R1
                J22 = R12*x[1] + R02*x[0] + 2.*R22*x[2] + R2
                return np.array([[J00, J01, J02],[J10, J11, J12],[J20, J21, J22]])

            init_vector  = np.array([1.0E-6, 1.0E-12, 1.0E-10])
            #results      = newton_krylov(cost, init_vector)
            results_root = root(cost, init_vector, jac = Jacobian, method = 'hybr')
            if not results_root.success:
                raise UserWarning("Solving of coupled ionization equilibrium equations did not succeed! Please check the results for sanity.")
            nHI   = results_root.x[0]
            nHeI  = results_root.x[1]
            nHeII = results_root.x[2]
            return nHI, nHeI, nHeII
        
        GcHI    = self.Gamma_c_HI(T)
        GcHeI   = self.Gamma_c_HeI(T)
        GcHeII  = self.Gamma_c_HeII(T)
        
        ArHII   = self.Alpha_r_HII(T)
        ArHeII  = self.Alpha_r_HeII(T)
        ArHeIII = self.Alpha_r_HeIII(T)
        AdHeII  = self.Alpha_d_HeIII(T)
        DHeII   = GcHeII + ArHeII + AdHeII

        if type(T) is float or type(T) is np.float_:
            GgH1  = self.HI_UV  
            GgHe1 = self.HeI_UV 
            GgHe2 = self.HeII_UV 
            nHI, nHeI, nHeII = eval(A,B,GcHI,GcHeI,GgH1,GgHe1,GgHe2,ArHII,ArHeII,ArHeIII,AdHeII,DHeII)

        elif type(T) is np.ndarray:
            GgH1  = np.ones(T.shape) * self.HI_UV  #1.0E-12 
            GgHe1 = np.ones(T.shape) * self.HeI_UV #1.0E-13 
            GgHe2 = np.ones(T.shape) * self.HeII_UV #1.0E-15 
            nHI = np.ones_like(T); nHeI = np.ones_like(T); nHeII = np.ones_like(T)
            with np.nditer(T, flags = ['multi_index']) as it:
                for i in it:
                    m = it.multi_index
                    nHI[m], nHeI[m], nHeII[m] = eval(A[m],B[m],GcHI[m],GcHeI[m],GgH1[m],GgHe1[m],GgHe2[m],ArHII[m],ArHeII[m],ArHeIII[m],AdHeII[m],DHeII[m])
        
        xHI  = nHI / nH; xHII = 1 - xHI
        xHeI = nHeI / nHe; xHeII = nHeII / nHe; xHeIII = 1 - xHeI - xHeII

        self.nH  = nH 
        self.nHe = nHe
        self.fractions = {'x_HI': xHI, 'x_HII': xHII, 'x_HeI': xHeI, 'x_HeII': xHeII, 'x_HeIII': xHeIII}
        return self.fractions






class Hydrogen:
    """
    Functionalities related to estimation of quantities pertaining to various species of Hydrogen (H). 
    This assumes that the baryon content of the Universe comprises fully of H and He (Helium).
    Wraps some functionalities of IonizationEquilibrium.
    """
    def __init__(self, Cosmology, X = 0.75, TREECOOL = None):
        """
        Initialize with the underlying Cosmological parameters, Hydrogen abundance, and a UVB photoionization rates table

        Args:
            Cosmology (dict):       dictionary of the cosmological parameters, 'h', 'Omega_baryons', 'Omega_matter', 'Omega_Lambda', 'redshift'
            X (float, optional):    mass abundance (fraction) of Hydrogen, default: 0.75 
            TREECOOL (str, optional):   full path to the TREECOOL UVB table file 
        """
        self.X          = X
        self.Y          = 1 - X
        self.IonEq      = IonizationEquilibrium(Cosmology, self.X, self.Y, TREECOOL = TREECOOL)
        self.H_0        = self.IonEq.H_0
        self.z          = self.IonEq.z
        self.Omega_b    = self.IonEq.Omega_b
        self.Omega_m    = self.IonEq.Omega_m
        self.Omega_L    = self.IonEq.Omega_L
        self.Hz 	    = self.IonEq.Hz
        self.critical_density = self.IonEq.critical_density
        self.Gamma_UVB  = self.IonEq.HI_UV # homogeneous ultraviolet background (UVB) photoionisation rate of HI
        self.Alpha      = self.IonEq.Alpha_r_HII 
        self.Gamma_coll = self.IonEq.Gamma_c_HI


    def eval_nH(self, rho_b):
        """
        Estimate the total number density (in CGS) of all Hydrogen species, given a baryon overdensity rho_b

        Args:
            rho_b (float [ndarray]): value of local baryon overdensity, i.e., (density / mean_density)

        Returns: 
            float [ndarray]: number density of H in cm^-3
        """
        try:
            nH    = self.IonEq.nH
        except:
            Rho_b = rho_b * self.Omega_b * self.critical_density * (1 + self.z)**3
            nH    = Rho_b * self.X / mH 
        return nH
    

    def eval_HI_approximately(self, rho_b, T):
        """
        Estimate the number density (in CGS) and mass fraction of neutral Hydrogen, given a baryon overdensity and a gas temperature.
        This is under the approximation that (at this appropriately low redshift) Helium is fully doubly ionized, i.e., x_HeIII = 1.

        Args:
            rho_b (float [ndarray]):  value of local baryon overdensity, i.e., (density / mean_density)
            T (float [ndarray]):      value of local gas temperature (in K)

        Returns: 
            nHI (float [ndarray]): number density of HI in cm^-3
            xHI (float [ndarray]): mass fraction of HI
        """
        A   = self.eval_nH(rho_b)
        B   = A * (1 - self.X)/(2.*self.X)
        G_e = self.Gamma_coll(T)
        A_r = self.Alpha(T)
        a   = G_e + A_r
        b   = -(G_e*(A+B) + A_r*(2.*A+B) + np.ones(T.shape)*self.Gamma_UVB) 
        c   = A_r * A * (A + B)
        D   = b**2 - 4.*a*c
        nHI = (-b - np.sqrt(D))/(2.*a)
        nH  = A
        xHI = nHI / nH
        return nHI, xHI 


    def eval_HI(self, rho_b, T, mode = 'iterative', n_iter_iterative = 2):
        """
        Estimate the number density (in CGS) and mass fraction of neutral Hydrogen, given a baryon overdensity and a gas temperature.

        Args:
            rho_b (float [ndarray]):  value of local baryon overdensity, i.e., (density / mean_density)
            T (float [ndarray]):      value of local gas temperature (in K)
            approximate (bool):       whether to use the approximation x_HeIII = 1 (faster if True) [Note: this flag wraps Hydrogen.eval_HI_approximately]

        Returns: 
            nHI (float [ndarray]): number density of HI in cm^-3
            xHI (float [ndarray]): mass fraction of HI
        """
        if mode == 'approximate':
            return self.eval_HI_approximately(rho_b, T)
        elif mode == 'iterative':
            fractions = self.IonEq.eval_fractions_iteratively(rho_b, T, n_iter = n_iter_iterative)
            xHI       = fractions['x_HI']
            nH        = self.IonEq.nH
            return xHI*nH, xHI
        
        elif mode == 'full':
            fractions = self.IonEq.eval_fractions(rho_b, T)
            xHI = fractions['x_HI']
            nH  = self.IonEq.nH
            return xHI*nH, xHI
        else:
            raise ValueError("Invalid mode for Hydrogen.eval_HI(). Use 'approximate', 'iterative', or 'full'.")




class Helium:
    """
    Functionalities related to estimation of quantities pertaining to various species of Helium (He).
    This assumes that the baryon content of the Universe comprises fully of Hydrogen (H) and He.
    Wraps some functionalities of IonizationEquilibrium.
    """
    def __init__(self, Cosmology, Y = 0.25, TREECOOL = None):
        """
        Initialize with the underlying Cosmological parameters, Helium abundance, and a UVB photoionization rates table

        Args:
            Cosmology (dict):       dictionary of the cosmological parameters, 'h', 'Omega_baryons', 'Omega_matter', 'Omega_Lambda', 'redshift'
            Y (float, optional):    mass abundance (fraction) of Helium, default: 0.25
            TREECOOL (str, optional): full path to the TREECOOL UVB table file 
        """
        self.X          = 1 - Y
        self.Y          = Y
        self.IonEq      = IonizationEquilibrium(Cosmology, self.X, self.Y, TREECOOL = TREECOOL)
        self.H_0        = self.IonEq.H_0
        self.z          = self.IonEq.z
        self.Omega_b    = self.IonEq.Omega_b
        self.Omega_m    = self.IonEq.Omega_m
        self.Omega_L    = self.IonEq.Omega_L
        self.Hz 	    = self.IonEq.Hz
        self.critical_density = self.IonEq.critical_density
        

    def eval_nHe(self, rho_b):
        """
        Estimate the total number density (in CGS) of all Helium species, given a baryon overdensity rho_b

        Args:
            rho_b (float [ndarray]): value of local baryon overdensity, i.e., (density / mean_density)

        Returns: 
            float [ndarray]: number density of He in cm^-3
        """
        try:
            nHe = self.IonEq.nHe
        except:
            Rho_b = rho_b * self.Omega_b * self.critical_density * (1 + self.z)**3
            nHe   = Rho_b * self.Y / (4. *mH) 
        return nHe
    

    def eval_HeI(self, rho_b, T):
        """
        Estimate the number density (in CGS) and mass fraction of neutral Helium (HeI), given a baryon overdensity and a gas temperature.

        Args:
            rho_b (float [ndarray]):  value of local baryon overdensity, i.e., (density / mean_density)
            T (float [ndarray]):      value of local gas temperature (in K)

        Returns: 
            nHeI (float [ndarray]): number density of HeI in cm^-3
            xHeI (float [ndarray]): mass fraction of HeI
        """
        try:
            fractions = self.IonEq.fractions
        except:
            fractions = self.IonEq.eval_fractions(rho_b, T)
        xHeI          = fractions['x_HeI']
        nHe           = self.IonEq.nHe
        return xHeI*nHe, xHeI
    

    def eval_HeII(self, rho_b, T):
        """
        Estimate the number density (in CGS) and mass fraction of singly-ionized Helium (HeII), given a baryon overdensity and a gas temperature.

        Args:
            rho_b (float [ndarray]):  value of local baryon overdensity, i.e., (density / mean_density)
            T (float [ndarray]):      value of local gas temperature (in K)

        Returns: 
            nHeII (float [ndarray]): number density of HeII in cm^-3
            xHeII (float [ndarray]): mass fraction of HeII
        """
        try:
            fractions = self.IonEq.fractions
        except:
            fractions = self.IonEq.eval_fractions(rho_b, T)
        xHeII         = fractions['x_HeII']
        nHe           = self.IonEq.nHe
        return xHeII*nHe, xHeII
    



class Lyman:
    """
    Functionalities related to estimation of quantities pertaining to the intergalactic Lyman-series absorption for skewers extracted from a cosmological hydrodynamic simulation.
    Wraps some functionalities of IonizationEquilibrium and Hydrogen.
    """
    def __init__(self, hfilepath, X = 0.75, Y = 0.25, TREECOOL = None):
        """
        Initialize with an input skewers file, Hydrogen and Helium abundances, and a UVB photoionization rates table

        Args:
            hfilepath (pathlike):   full path to the input skewers file
            X (float, optional):    mass abundance (fraction) of Hydrogen, default: 0.75`
            Y (float, optional):    mass abundance (fraction) of Helium, default: 0.25
            TREECOOL (str, optional): full path to the TREECOOL UVB table file 
        """
        self.hfilepath = hfilepath
        self.Cosmology, self.Domain = cosmology(hfilepath)
        self.IonEq     = IonizationEquilibrium(self.Cosmology, X = X, Y = Y, TREECOOL = TREECOOL)
        self.Hydrogen = Hydrogen(self.Cosmology, X = X, TREECOOL = TREECOOL)
        self.Helium   = Helium(self.Cosmology, Y = Y, TREECOOL = TREECOOL)
        self.H0       = self.IonEq.H_0
        self.z        = self.IonEq.z
        self.Hz       = self.IonEq.Hz
        self. X       = X
        self. Y       = Y
        f_lu_path     = os.path.join(cwd, "oscillator_strengths.yml")
        lammbda0_path = os.path.join(cwd, "rest_wavelengths.yml")
        with open(f_lu_path, 'r') as f:
            self.f_lu_table = yaml.safe_load(f)
        with open(lammbda0_path, 'r') as f:
            self.lambda_0_table = yaml.safe_load(f)


    def eval_tau_local(self, rho_b, T, n_neutral = None, element = 'H', transition = 'lya', nH1_mode = 'approximate', n_iter_iterative = 2):
        """
        Estimate the local optical depth of the given resonant Lyman series attenuation for the given element, from an absorber's baryon overdensity and gas temperature.

        Args:
            rho_b (float [ndarray]):  value of local baryon overdensity, i.e., (density / mean_density)
            T (float [ndarray]):      value of local gas temperature (in K)
            n_neutral (float [ndarray]):   number density of neutral atoms, if provided, overrides the calculation from rho_b and T. Supply this if you already have the number density of neutral atoms for the given element for avoiding recomputation.
            element (str):            element for which to estimate the resonant optical depth, options: 'H', 'He'
            transition (str):         transition for which to estimate the resonant optical depth, options: 'lya', 'lyb'
            nH1_mode (str):           method to use for estimating the number density of HI, options: 'iterative', 'approximate', 'full'
            n_iter_iterative (int):   number of iterations to perform if choosing 'iterative' for `nH1_mode`

        Returns: 
            tau (float [ndarray]):    local resonant optical depth
        """
        f_lu      = self.f_lu_table[f'{element}_{transition}']
        lambda_0  = self.lambda_0_table[f'{element}_{transition}'] # in Angstroms
        C0        = np.pi * q_e**2 * f_lu * (lambda_0 * 1E-8) / (m_e * c_light * self.Hz)
        if n_neutral is None:
            if element == 'H':
                n_neutral  = self.Hydrogen.eval_HI(rho_b, T, mode = nH1_mode, n_iter_iterative = n_iter_iterative)[0]  # number density of HI in cm^-3
            elif element == 'He':
                n_neutral  = self.Helium.eval_HeI(rho_b, T)[0]
        tau_local = C0 * n_neutral
        return tau_local


    def eval_tau_skewer(self, rho_b, T, v_pec_los, n_neutral = None, element = 'H', transition = 'lya', profile = 'doppler', n_pix_out = None, nH1_mode = 'approximate', n_iter_iterative = 2, return_v_h_skewer = False):
        """
        Estimate the optical depth of the given resonant Lyman series attenuation for the given element as a spectrum, from an absorber's baryon overdensity, gas temperature and the LOS component of its peculiar velocity along a skewer.

        Args:
            rho_b (ndarray):          baryon density along the skewer in units of mean baryon density, i.e., (density / mean_density)
            T (ndarray):              gas temperature along the skewer (in K)
            v_pec_los (ndarray):      parallel component of the peculiar velocity along the LOS (in km/s)
            n_neutral (float [ndarray]):   number density of neutral atoms, if provided, overrides the calculation from rho_b and T. Supply this if you already have the number density of neutral atoms for the given element for avoiding recomputation.
            element (str):            element for which to estimate the resonant optical depth, options: 'H', 'He'
            transition (str):         transition for which to estimate the resonant optical depth, options: 'lya', 'lyb'
            n_pix_out (int):          number of pixels in the output spectrum (we have infinite spectral resolution, so its pixel-limited), if None, uses the length of v_pec_los
            profile (str):            type of line profile to use, options: 'erf', 'doppler', 'voigt' ('erf' refers to the error function approximation of the Doppler profile for discrete pixels)
            nH1_mode (str):           method to use for estimating the number density of HI, options: 'iterative', 'approximate', 'full'
            n_iter_iterative (int):   number of iterations to perform if choosing 'iterative' mode for nH1_mode
            return_v_h_skewer (bool): whether to return the Hubble flow velocity of the output spectrum, default: False

        Returns: 
            tau (ndarray):            resonant optical depth
            v_h_skewer (ndarray):     Hubble flow velocity of the output spectrum (in km/s), if return_v_h_skewer is True
        """
        f_lu     = self.f_lu_table[f'{element}_{transition}']
        lambda_0 = self.lambda_0_table[f'{element}_{transition}'] # in Angstroms
        C0 = np.pi * q_e**2 * f_lu * (lambda_0 * 1E-8) / (m_e * c_light * self.Hz)     # overall constant factor
        if n_neutral is None:
            if element == 'H':
                n_neutral  = self.Hydrogen.eval_HI(rho_b, T, mode = nH1_mode, n_iter_iterative = n_iter_iterative)[0]    # number density of HI in cm^-3
            elif element == 'He':
                n_neutral  = self.Helium.eval_HeI(rho_b, T)[0]

        if n_pix_out is None:
            n_pix_out = len(v_pec_los)

        profile       = profile.lower()
        sigma_Doppler = np.sqrt(kB * T / mH)
        gamma_Lorentz = 2. * np.pi * q_e**2 * f_lu / (3. * m_e * c_light * (lambda_0 * 1E-8)) 

        frac_extend   = 0.25
        len_extend    = int(frac_extend * len(v_pec_los))
        Hubble_hfile  = Hubble(self.hfilepath)
        v_h_skewer    = Hubble_hfile.v_hubble(np.arange(n_pix_out) + len_extend) * (len(v_pec_los) / n_pix_out)  # Hubble flow velocity of the output spectrum in cm/s

        v_pec_los *= 1E5  # convert to cm/s
        vlos_sk_extended = np.concatenate(
            [v_pec_los[-len_extend:], v_pec_los, v_pec_los[:len_extend]]
        )
        sigma_Doppler_extended = np.concatenate(
            [sigma_Doppler[-len_extend:], sigma_Doppler, sigma_Doppler[:len_extend]]
        )
        n_neutral_extended = np.concatenate(
            [n_neutral[-len_extend:], n_neutral, n_neutral[:len_extend]]
        )
        v_h_skewer_input_extended = Hubble_hfile.v_hubble(
            np.arange(len(vlos_sk_extended))
        )
        v_total_input_extended = v_h_skewer_input_extended + vlos_sk_extended
        # dv_total_cells         = v_total_input_extended[1:] - v_total_input_extended[:-1]
        dv_cells = np.diff(v_h_skewer_input_extended)[0]  # assume uniform spacing in the input skewer
        # v_total_input_extended = v_total_input_extended[:-1]
        Profile = LineProfile(v_total_input_extended, sigma_Doppler_extended, gamma_Lorentz, dv_cells = dv_cells) #dv_total_cells) 
        if profile == 'erf':
            profile_func = lambda v: Profile.error_function(v)/ dv_cells #np.diff(v_h_skewer_input_extended)[0]
        elif profile == 'doppler':
            profile_func = Profile.doppler
        elif profile == 'voigt':
            profile_func = Profile.voigt
        elif profile == 'interp_doppler':
            profile_func = Profile.create_interpolator_doppler(5,5000)
        elif profile == 'interp_voigt':
            Nsig = 250
            pf = Profile.create_interpolator_voigt(Nsig, Profile.sigma.min(), Profile.sigma.max(), 500, 30)
            def profile_func(v):
                f = pf(v)
                f[np.abs((v-v_total_input_extended)/Profile.sigma)>Nsig] = 0
                return f
        tau = []
        for j in range(n_pix_out):
            profile_extended = profile_func(v_h_skewer[j])
            tau_j = C0 * np.sum(n_neutral_extended * profile_extended) * np.diff(v_h_skewer_input_extended)[0]
            tau.append(tau_j)
        tau = np.array(tau)
        if return_v_h_skewer:
            return tau, (v_h_skewer - v_h_skewer[0]) * 1E-5  # convert to km/s and return the Hubble flow velocity of the output spectrum
        else:
            return tau