import numpy as np

class Bachelier:

    def __init__(self,S0 : float, r : float, sigma : float, q : float):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.q = q



    def compute_d1_d2(self, strike : float, t : float, T : float):
        d1 = 1/(self.sigma * np.sqrt(T-t)) *(np.log(self.S0/strike) + (self.r - self.q + (self.sigma**2)/2 )*(T-t) )
        d2 = d1 - self.sigma*np.sqrt(T - t)