import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
import FinancialInstrument as fin_instr

class Option(fin_instr.FinancialInstrument):

    def __init__(self, payoff : Callable,S0 : float, T : float):
        super().__init__(T = T)
        self.payoff = payoff
        self.S0 = S0



