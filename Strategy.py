import numpy as np
import matplotlib.pyplot as plt
import EuropeanOption as euro


class OptionStrat:
    def __init__(self, products : list, position : list):
        self.products = products
        self.position = position



class CallSpread(OptionStrat):

    def __init__(self,model, K1 : float, K2 : float, S0 : float,t : float, T : float, r : float, q : float, sigma : float, pos :str = "long"):
        call1 = euro.EuropeanCallOption(S0 = S0, K = K1,t = t, T = T, model = model, r = r, q = q, sigma = sigma)
        call2 = euro.EuropeanCallOption(S0 = S0, K = K2,t = t, T = T, model = model, r = r, q = q, sigma = sigma)
        if(pos == "long"):
            pos_temp = [1,-1]
        elif (pos == "short"):
            pos_temp = [-1,1]
        super().__init__(products = [call1,call2], position = pos_temp)



    def plot_payoff(self):
        final_payoff = lambda STOCK_PRICE, STRIKE1, STRIKE2 : self.position[0]*self.products[0].payoff(STOCK_PRICE, STRIKE1) + self.position[1]*self.products[1].payoff(STOCK_PRICE, STRIKE2)
        S0 = self.products[0].model.S0
        K_min = np.minimum(self.products[0].K,self.products[1].K)
        K_max = np.maximum(self.products[0].K,self.products[1].K)
        x_min = np.minimum(S0, K_min) / 2
        x_max = np.maximum(S0, K_max) * 4 / 3
        n_points = int(np.maximum(x_max - x_min, 100) + 1)
        S0_arr = np.linspace(x_min, x_max, n_points)

        plt.plot(S0_arr, final_payoff(STOCK_PRICE=S0_arr, STRIKE1 = self.products[0].K, STRIKE2 = self.products[1].K))
        plt.grid() ; plt.xlabel("stock price")
        plt.title(f"payoff CALL-SPREAD : {self.products[0].K}-{self.products[1].K}")
        plt.show()

    def plot_PNL(self):
        final_payoff = lambda STOCK_PRICE, STRIKE1, STRIKE2: self.position[0] * self.products[0].payoff(STOCK_PRICE, STRIKE1) + self.position[1] * self.products[1].payoff(STOCK_PRICE, STRIKE2)

        S0 = self.products[0].model.S0
        price_call1 = self.products[0].compute_price()*self.position[0]*(-1)
        price_call2 = self.products[1].compute_price()*self.position[1]*(-1)
        price_strat = price_call1 + price_call2

        K_min = np.minimum(self.products[0].K, self.products[1].K)
        K_max = np.maximum(self.products[0].K, self.products[1].K)
        x_min = np.minimum(S0, K_min) / 2
        x_max = np.maximum(S0, K_max) * 4 / 3
        n_points = int(np.maximum(x_max - x_min, 200) + 1)
        S0_arr = np.linspace(x_min, x_max, n_points)


        PNL_call1 = price_call1 + self.position[0]*self.products[0].payoff(STOCK_PRICE = S0_arr,STRIKE = self.products[0].K )
        PNL_call2 = price_call2 + self.position[1] * self.products[1].payoff(STOCK_PRICE=S0_arr, STRIKE=self.products[1].K)
        PNL_strat = price_strat + final_payoff(STOCK_PRICE=S0_arr, STRIKE1=self.products[0].K, STRIKE2=self.products[1].K)
        indx_positiv_pnl = np.where(PNL_strat >= 0)[0]

        fig, ax = plt.subplots()
        ax.plot(S0_arr, PNL_call1 ,label=f"long CALL | K = {self.products[0].K}", alpha = 0.7, linestyle = '--')
        ax.plot(S0_arr, PNL_call2 ,label=f"short CALL | K = {self.products[1].K}", alpha = 0.7, linestyle = '--')
        ax.plot(S0_arr, PNL_strat , label = "call spread", alpha = 1, color = "blue")

        ax.axvline(x=self.products[0].S0, color = "grey", label = "$S_0$", linestyle = "--")

        ax.fill_between(S0_arr, PNL_strat, where =PNL_strat > 0, facecolor='green', alpha=0.5)
        ax.fill_between(S0_arr, PNL_strat, where= PNL_strat <= 0, facecolor='red', alpha=0.5)

        plt.grid(); plt.legend()
        plt.ylabel("P&L")
        plt.xlabel("stock price")
        plt.title(f"P&L CALL-SPREAD : {self.products[0].K}-{self.products[1].K}")
        plt.show()
