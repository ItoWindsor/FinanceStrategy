import numpy as np

class BlackScholes:

    def __init__(self,S0 : float, r : float, sigma : float, q : float):
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.q = q



    def compute_d1_d2(self, strike : float, t : float, T : float, S0_arr : np.array = None):
        if S0_arr is not None:
            d1 = 1 / (self.sigma * np.sqrt(T - t)) * (np.log(S0_arr / strike) + (self.r - self.q + (self.sigma ** 2) / 2) * (T - t))
        else:
            d1 = 1/(self.sigma * np.sqrt(T-t)) *(np.log(self.S0/strike) + (self.r - self.q + (self.sigma**2)/2 )*(T-t) )

        d2 = d1 - self.sigma*np.sqrt(T - t)
        return d1,d2