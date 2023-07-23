import numpy as np

PAYOFF_CALL = lambda STOCK_PRICE,STRIKE : np.maximum(STOCK_PRICE - STRIKE,0)
PAYOFF_PUT = lambda STOCK_PRICE,STRIKE : np.maximum(STRIKE - STOCK_PRICE,0)