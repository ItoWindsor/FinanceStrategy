import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
import model as mo

class Option:

    def __init__(self,model : str, payoff : Callable,S0 : float, t : float, T : float, r : float, q : float, sigma : float):
        if (model == "BS"):
            self.model = mo.BlackScholes(S0,r = r, sigma = sigma, q = q)
        self.t = t
        self.T = T
        self.payoff = payoff
        self.S0 = S0



