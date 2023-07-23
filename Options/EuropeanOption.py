import numpy as np
import matplotlib.pyplot as plt
from typing import Callable

import Option as op
import UsualPayoffs

class EuropeanOption(op.Option):

    def __init__(self,payoff : Callable,S0 : float, T : float):
        super().__init__(payoff = payoff, S0 = S0, T = T)




class EuropeanCallOption(EuropeanOption):

    def __init__(self,S0: float, K : float, T : float, closed_formula : bool = True):
        super().__init__(payoff = UsualPayoffs.PAYOFF_CALL, S0 = S0, T = T)
        self.K = K

    def compute_price(self):
        return(-1)
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

    def __init__(self,S0: float, K : float, T : float):
        super().__init__(payoff = UsualPayoffs.PAYOFF_PUT, S0 = S0, T = T)
        self.K = K

    def plot_payoff(self):
        x_min = np.minimum(self.S0,self.K)/2
        x_max = np.maximum(self.S0,self.K)*4/3
        n_points = int(np.maximum(x_max - x_min, 100)+1)
        S0_arr = np.linspace( x_min, x_max, n_points)

        plt.plot(S0_arr, self.payoff(STOCK_PRICE = S0_arr, STRIKE = self.K))
        plt.grid()
        plt.title(f"payoff PUT : K = {self.K}")
        plt.show()