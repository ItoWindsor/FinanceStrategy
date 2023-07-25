import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
import model as mo

class Option:

    def __init__(self,model : str, payoff : Callable,S0 : float, t : float, T : float, r : float, q : float, sigma : float, name : str = None):
        if model == "BS":
            self.model = mo.BlackScholes(S0,r = r, sigma = sigma, q = q)
        self.t = t
        self.T = T
        self.payoff = payoff
        self.S0 = S0
        self.name = name

    def compute_features(self, features : tuple = ("price",)) -> dict:
        dic_features = {"S0": self.S0}
        for item in features:
            match item:
                case "price":
                    dic_features[item] = self.compute_price()
                case "delta":
                    dic_features[item] = self.compute_delta()
                case "gamma":
                    dic_features[item] = self.compute_gamma()
                case "rho":
                    dic_features[item] = self.compute_rho()
                case "vega":
                    dic_features[item] = self.compute_vega()

        return dic_features
        pass

    def compute_price(self) -> float:
        pass

    def compute_delta(self) -> float:
        pass

    def compute_gamma(self) -> float:
        pass

    def compute_rho(self) -> float:
        pass

    def compute_vega(self) -> float:
        pass






