import numpy as np
import matplotlib.pyplot as plt
import EuropeanOption as euro

dic_pos = {"long" : -1,
           "short" : 1}

def sum_payoffs(f1,f2, STRIKE_input,pos_f2):
    return lambda STOCK_PRICE : f1(STOCK_PRICE = STOCK_PRICE) + pos_f2*f2(STOCK_PRICE = STOCK_PRICE, STRIKE = STRIKE_input)*(-1)

class OptionStrat:
    def __init__(self, products : np.array = np.empty(shape = 0) , position : np.array = np.empty(shape = 0), strat_name : str = None):
        self.products = products
        self.position = position
        self.strat_name = strat_name

    def final_payoff(self):
        f = lambda STOCK_PRICE : self.products[0].payoff(STOCK_PRICE = STOCK_PRICE, STRIKE = self.products[0].K)
        for i in range(1,len(self.products)):
            f = sum_payoffs(f1= f,f2 = self.products[i].payoff,STRIKE_input= self.products[i].K, pos_f2 = dic_pos[self.position[i]])
        return f

    def add_products(self, prod , pos : str):
        self.products = np.append(self.products, prod)
        self.position = np.append(self.position, pos)

    def plot_PNL(self):
        FINAL_PAYOFF = self.final_payoff()

        S0 = self.products[0].model.S0
        K_min = self.products[0].K
        K_max = self.products[0].K
        for i in range(1,len(self.products)):
            K_min = np.minimum(K_min, self.products[i].K)
            K_max = np.maximum(K_max, self.products[i].K)
        x_min = np.minimum(S0, K_min) / 2
        x_max = np.maximum(S0, K_max) * 4 / 3
        n_points = int(np.maximum(x_max - x_min, 100) + 1)
        S0_arr = np.linspace(x_min, x_max, n_points)

        price_products = []
        price_strat = 0
        for i in range(len(self.products)):
            price_products.append(self.products[i].compute_price() * dic_pos[self.position[i]])
            price_strat += price_products[-1]
        PNL_strat = price_strat + FINAL_PAYOFF(STOCK_PRICE=S0_arr)


        fig, ax = plt.subplots()
        for i in range(len(self.products)):
            ax.plot(S0_arr, price_products[i] + dic_pos[self.position[i]]*self.products[i].payoff(STOCK_PRICE = S0_arr, STRIKE = self.products[i].K)*(-1),
                    label=f"{self.position[i]} {self.products[i]} | K = {self.products[i].K}", alpha=0.7, linestyle='--')

        ax.axvline(x=self.products[0].S0, color="grey", label="$S_0$", linestyle="--")
        ax.plot(S0_arr, PNL_strat, label="custom strat", alpha=1, color="blue")

        ax.fill_between(S0_arr, PNL_strat, where=PNL_strat > 0, facecolor='green', alpha=0.5)
        ax.fill_between(S0_arr, PNL_strat, where=PNL_strat <= 0, facecolor='red', alpha=0.5)

        plt.grid(); plt.legend()
        plt.ylabel("P&L"); plt.xlabel("stock price")

        if self.strat_name is None:
            title_str = "P&L custom strat : "
        else:
            title_str = f"P&L {self.strat_name} : "

        for i in range(len(self.products)):
            title_str += f"{self.products[i].K}"
            if i < len(self.products)-1:
                title_str += "-"
        plt.title(title_str)
        plt.show()

    def plot_current_PNL(self):
        FINAL_PAYOFF = self.final_payoff()

        S0 = self.products[0].model.S0
        K_min = self.products[0].K
        K_max = self.products[0].K
        for i in range(1, len(self.products)):
            K_min = np.minimum(K_min, self.products[i].K)
            K_max = np.maximum(K_max, self.products[i].K)
        x_min = np.minimum(S0, K_min) / 2
        x_max = np.maximum(S0, K_max) * 4 / 3
        n_points = int(np.maximum(x_max - x_min, 300) + 1)
        S0_arr = np.linspace(x_min, x_max, n_points)

        price_products = []
        price_strat = 0
        for i in range(len(self.products)):
            price_products.append(self.products[i].compute_price() * dic_pos[self.position[i]])
            price_strat += price_products[-1]
        PNL_strat = price_strat + FINAL_PAYOFF(STOCK_PRICE=S0_arr)

        current_pnl = np.zeros(shape=S0_arr.shape)
        for i in range((len(self.products))):
            current_pnl += self.products[i].compute_price(S0_arr = S0_arr) * dic_pos[self.position[i]]*(-1)

        current_pnl +=  price_strat


        fig, ax = plt.subplots()
        ax.axvline(x=self.products[0].S0, color="grey", label="$S_0$", linestyle="--")
        ax.plot(S0_arr, PNL_strat, label="final P&L", alpha=1, color="blue")
        ax.plot(S0_arr, current_pnl, label = "current P&L", alpha = 1, color = "brown")
        ax.fill_between(S0_arr, PNL_strat, where=PNL_strat > 0, facecolor='green', alpha=0.5)
        ax.fill_between(S0_arr, PNL_strat, where=PNL_strat <= 0, facecolor='red', alpha=0.5)

        plt.grid(); plt.legend() ; plt.ylabel("P&L");  plt.xlabel("stock price")

        if self.strat_name is None:
            title_str = "P&L custom strat : "
        else:
            title_str = f"P&L {self.strat_name} : "

        for i in range(len(self.products)):
            title_str += f"{self.products[i].K}"
            if i < len(self.products) - 1:
                title_str += "-"
        title_str += f" | time to maturity : {self.products[0].T - self.products[0].t}"
        plt.title(title_str)
        plt.show()


    def compute_features(self, features = ('price',)) -> dict:
        dic_features = {"S0": self.products[0].S0}
        for feature in features:
            dic_features[feature] = 0

        for item in features:
            match item:
                case "price":
                    for prod,pos in zip(self.products,self.position):
                        dic_features["price"] += prod.compute_price()*dic_pos[pos]*(-1)
                case "delta":
                    for prod in self.products:
                        dic_features["delta"] += prod.compute_delta()*dic_pos[pos]*(-1)
                case "gamma":
                    for prod in self.products:
                        dic_features["gamma"] += prod.compute_gamma()*dic_pos[pos]*(-1)
                case "rho":
                    for prod in self.products:
                        dic_features["rho"] += prod.compute_rho()*dic_pos[pos]*(-1)
                case "vega":
                    for prod in self.products:
                        dic_features["vega"] += prod.compute_vega()*dic_pos[pos]*(-1)
        return dic_features

class CallSpread(OptionStrat):

    def __init__(self,model, K1 : float, K2 : float, S0 : float,t : float, T : float, r : float, q : float, sigma : float, pos :str = "long"):
        call1 = euro.EuropeanCallOption(S0 = S0, K = K1,t = t, T = T, model = model, r = r, q = q, sigma = sigma)
        call2 = euro.EuropeanCallOption(S0 = S0, K = K2,t = t, T = T, model = model, r = r, q = q, sigma = sigma)
        if pos == "long":
            pos_temp = ["long","short"]
        elif pos == "short":
            pos_temp = ["short","long"]
        super().__init__(products = [call1,call2], position = pos_temp)



    def plot_payoff(self):
        final_payoff = lambda STOCK_PRICE, STRIKE1, STRIKE2 : dic_pos[self.position[0]]*self.products[0].payoff(STOCK_PRICE, STRIKE1)*(-1) + dic_pos[self.position[1]]*self.products[1].payoff(STOCK_PRICE, STRIKE2)*(-1)
        S0 = self.products[0].model.S0
        K_min = np.minimum(self.products[0].K,self.products[1].K)
        K_max = np.maximum(self.products[0].K,self.products[1].K)
        x_min = np.minimum(S0, K_min) / 2
        x_max = np.maximum(S0, K_max) * 4 / 3
        n_points = int(np.maximum(x_max - x_min,300) + 1)
        S0_arr = np.linspace(x_min, x_max, n_points)

        plt.plot(S0_arr, final_payoff(STOCK_PRICE=S0_arr, STRIKE1 = self.products[0].K, STRIKE2 = self.products[1].K))
        plt.grid() ; plt.xlabel("stock price")
        plt.title(f"payoff CALL-SPREAD : {self.products[0].K}-{self.products[1].K}")
        plt.show()

    def plot_PNL(self):
        final_payoff = lambda STOCK_PRICE, STRIKE1, STRIKE2: dic_pos[self.position[0]] * self.products[0].payoff(STOCK_PRICE, STRIKE1)*(-1) + dic_pos[self.position[1]] * self.products[1].payoff(STOCK_PRICE, STRIKE2)*(-1)

        S0 = self.products[0].model.S0
        price_call1 = self.products[0].compute_price()*dic_pos[self.position[0]]
        price_call2 = self.products[1].compute_price()*dic_pos[self.position[1]]
        price_strat = price_call1 + price_call2

        K_min = np.minimum(self.products[0].K, self.products[1].K)
        K_max = np.maximum(self.products[0].K, self.products[1].K)
        x_min = np.minimum(S0, K_min) / 2
        x_max = np.maximum(S0, K_max) * 4 / 3
        n_points = int(np.maximum(x_max - x_min, 300) + 1)
        S0_arr = np.linspace(x_min, x_max, n_points)


        PNL_call1 = price_call1 + dic_pos[self.position[0]]*self.products[0].payoff(STOCK_PRICE = S0_arr,STRIKE = self.products[0].K )*(-1)
        PNL_call2 = price_call2 + dic_pos[self.position[1]]*self.products[1].payoff(STOCK_PRICE=S0_arr, STRIKE=self.products[1].K)*(-1)
        PNL_strat = price_strat + final_payoff(STOCK_PRICE=S0_arr, STRIKE1=self.products[0].K, STRIKE2=self.products[1].K)

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
