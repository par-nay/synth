import numpy as np

class Gamma_coll:
    """
    Collisional ionization rates from Voronov (1997)
    """
    def __init__(self, m, A, P, E, X):
        self.m = m
        self.A = A
        self.P = P
        self.E = E
        self.X = X
        
    def fit(self, T):
        """
        Estimate the collisional ionization rate of the given species for a certain temperature (T/K)

        Args:
            T (float [ndarray]): gas temperature in K

        Returns:
            float [ndarray]: Collisional ionization rate in CGS (cm^3 s^-1)
        """
        U = 11604.5 * self.E / T
        f = self.A * ((1. + self.P * np.sqrt(U)) / (self.X + U)) * (U)**(self.m) * np.exp(-U)
        return f #* 1.0E-6 #in m^3 s^-1
	

class Alpha_r:
    """
    Recombination rates from Verner & Ferland (1996)
    """
    def __init__(self, a, b, T0, T1):
        self.a  = a
        self.b  = b
        self.T0 = T0
        self.T1 = T1 
        
    def fit(self, T):
        """
        Estimate the recombination rate of the given species for a certain temperature (T/K)

        Args:
            T (float [ndarray]): gas temperature in K

        Returns:
            float [ndarray]: Recombination rate in CGS (cm^3 s^-1)
        """
        f       = self.a * ( np.sqrt(T/self.T0) * (1. + np.sqrt(T/self.T0))**(1 - self.b) * (1. + np.sqrt(T/self.T1))**(1 + self.b) )**(-1.)
        return f #* 1.0E-6 #in m^3 s^-1
    
    def alternative(self, T, a1, b1, T01, T11, threshold = 1.0E6):
        """
        Estimate the recombination rate of the given species for an array of temperature values (T/K) with a temperature threshold 

        Args:
            T (ndarray): gas temperatures in K
            a1, b1, T01, T11 (floats): alternative set of parameters of Verner & Ferland (1996)
            threshold (float, optional): threshold of temperature in K to split the two different parameter combinations of Verner & Ferland (1996), default: 10^6 K

        Returns:
            ndarray: Recombination rates in CGS (cm^3 s^-1)
        """
        g   = np.zeros(len(T))
        g[T<=threshold] = self.fit(T[T<=threshold])
        g[T>threshold]  = Alpha_r(a1, b1, T01, T11).fit(T[T>threshold])
        return g #* 1.0E-6 #in m^3 s^-1
	
	
def Alpha_d(T):
    """
    Dielectric recombination rate of singly ionized Helium (HeII) from Aldrovandi & Pequignot (1973)

    Args:
        T (float [ndarray]): gas temperature in K

    Returns:
        float [ndarray]: Dielectric recombination rate of HeII in CGS (cm^3 s^-1)
    """
    return 1.9E-3 * (1. + 0.3*np.exp( -9.4E4 / T)) * np.exp(-4.7E5 / T) * T**(-1.5) #* 1.0E-6 # in m^3 s^-1