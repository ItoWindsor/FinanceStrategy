import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
import scipy.stats as stat

import Option as op
import UsualPayoffs
import model as mo

class EuropeanOption(op.Option):

    def __init__(self,model : str, payoff : Callable,S0 : float, t : float, T : float, r : float, q : float, sigma : float, name : str = None):
        super().__init__(payoff = payoff,model = model, t = t, S0 = S0, T = T, r = r, q = q, sigma = sigma, name = name)


    def compute_price(self) -> float:
        pass
    def compute_rho(self) -> float:
        pass

    def compute_gamma(self) -> float:
        pass

    def compute_delta(self) -> float:
        pass

    def compute_vega(self) -> float:
        pass

class EuropeanCallOption(EuropeanOption):

    def __init__(self,model : str,S0 : float, K : float, t : float, T : float, r : float, q : float, sigma : float, closed_formula : bool = True):
        super().__init__(payoff = UsualPayoffs.PAYOFF_CALL,model = model, t = t, S0 = S0, T = T, r = r, q = q, sigma = sigma, name = "CALL")
        self.K = K
        self.closed_formula = closed_formula

    def __str__(self):
        return "CALL"

    def compute_price(self,S0_arr : np.array = None):
        if type(self.model) == mo.BlackScholes:
            d1,d2 = self.model.compute_d1_d2(strike = self.K, t = self.t, T = self.T, S0_arr = S0_arr)
            if S0_arr is None:
                price = self.S0*np.exp(-self.model.q * (self.T - self.t )) * stat.norm.cdf(d1) - self.K*np.exp(-self.model.r*(self.T - self.t )) * stat.norm.cdf(d2)
            else:
                price = S0_arr * np.exp(-self.model.q * (self.T - self.t)) * stat.norm.cdf(d1) - self.K * np.exp(-self.model.r * (self.T - self.t)) * stat.norm.cdf(d2)
            return price

    def compute_delta(self) -> float:
        if type(self.model) == mo.BlackScholes:
            d1,d2 = self.model.compute_d1_d2(strike = self.K, t = self.t, T = self.T)
            return stat.norm.cdf(d1)

    def compute_gamma(self) -> float:
        if type(self.model) == mo.BlackScholes:
            d1, d2 = self.model.compute_d1_d2(strike=self.K, t=self.t, T=self.T)
            return stat.norm.pdf(d1)/(self.S0 * self.model.sigma * np.sqrt(self.T - self.t))

    def compute_rho(self) -> float:
        if type(self.model) == mo.BlackScholes:
            d1, d2 = self.model.compute_d1_d2(strike=self.K, t=self.t, T=self.T)
            return self.K * (self.T - self.t) * np.exp(-self.model.r * (self.T - self.t)) * stat.norm.cdf(d2)

    def compute_vega(self) -> float:
        if type(self.model) == mo.BlackScholes:
            d1, d2 = self.model.compute_d1_d2(strike=self.K, t=self.t, T=self.T)
            return self.S0*stat.norm.pdf(d1)*np.sqrt(self.T - self.t)

    def plot_payoff(self):
        x_min = np.minimum(self.S0,self.K)/2
        x_max = np.maximum(self.S0,self.K)*4/3
        n_points = int(np.maximum(x_max - x_min, 100)+1)
        S0_arr = np.linspace( x_min, x_max, n_points)

        plt.plot(S0_arr, self.payoff(STOCK_PRICE = S0_arr, STRIKE = self.K))
        plt.grid()
        plt.title(f"payoff CALL : K = {self.K}")
        plt.show()

class EuropeanPutOption(EuropeanOption):

    def __init__(self,model : str, S0 : float, K : float, t : float, T : float, r : float, q : float, sigma : float, closed_formula : bool = True):
        super().__init__(payoff = UsualPayoffs.PAYOFF_PUT,model = model, t = t, S0 = S0, T = T, r = r, q = q, sigma = sigma, name = "PUT")
        self.K = K

    def __str__(self):
        return "PUT"
    def plot_payoff(self):
        x_min = np.minimum(self.S0,self.K)/2
        x_max = np.maximum(self.S0,self.K)*4/3
        n_points = int(np.maximum(x_max - x_min, 100)+1)
        S0_arr = np.linspace( x_min, x_max, n_points)

        plt.plot(S0_arr, self.payoff(STOCK_PRICE = S0_arr, STRIKE = self.K))
        plt.grid()
        plt.title(f"payoff PUT : K = {self.K}")
        plt.show()

    def compute_price(self, S0_arr : np.array = None):
        if type(self.model) == mo.BlackScholes:
            d1,d2 = self.model.compute_d1_d2(strike = self.K, t = self.t, T = self.T)
            if S0_arr is None:
                price =  self.K*np.exp(-self.model.r*(self.T - self.t )) * stat.norm.cdf(-d2) - self.S0*np.exp(-self.model.q * (self.T - self.t )) * stat.norm.cdf(-d1)
            else:
                price = self.K * np.exp(-self.model.r * (self.T - self.t)) * stat.norm.cdf(-d2) - S0_arr* np.exp( -self.model.q * (self.T - self.t)) * stat.norm.cdf(-d1)
            return price

    def compute_delta(self) -> float:
        if type(self.model) == mo.BlackScholes:
            d1,d2 = self.model.compute_d1_d2(strike = self.K, t = self.t, T = self.T)
            return - stat.norm.cdf(-d1)

    def compute_gamma(self) -> float:
        if type(self.model) == mo.BlackScholes:
            d1, d2 = self.model.compute_d1_d2(strike=self.K, t=self.t, T=self.T)
            return stat.norm.pdf(d1)/(self.S0 * self.model.sigma * np.sqrt(self.T - self.t))
    def compute_rho(self) -> float:
        if type(self.model) == mo.BlackScholes:
            d1, d2 = self.model.compute_d1_d2(strike=self.K, t=self.t, T=self.T)
            return -self.K*(self.T - self.t)*np.exp(-self.model.r*(self.T - self.t))*stat.norm.cdf(-d2)

    def compute_vega(self) -> float:
        if type(self.model) == mo.BlackScholes:
            d1, d2 = self.model.compute_d1_d2(strike=self.K, t=self.t, T=self.T)
            return self.S0 * stat.norm.pdf(d1) * np.sqrt(self.T - self.t)