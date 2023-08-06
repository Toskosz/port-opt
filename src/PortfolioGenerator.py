import pandas as pd
from datetime import datetime
import numpy as np
from cvxopt import matrix, solvers
import matplotlib.pyplot as plt


class PortfolioGenerator:
    
    def __init__(self) -> None:
        self.data = {}
        self.number_of_assets = None
        
        self.historical_mean = None
        self.historical_cov = None

        self.random_portfolios_returns = None
        self.random_portfolios_stdevs = None
        pass

    def load_assets_data(self, assets, path):
        for asset in assets:
            self.data[asset] = {}

        with open(path) as file:
            for line in file:
                try:
                    code = line[12:24].strip()
                    # If i just do data.get(code) python would evaluate {}
                    # as False
                    if self.data.get(code, 0) == 0:
                        continue
                    else:
                        bdi_code = line[10:12].strip()
                        if bdi_code != '02':
                            continue

                        price = line[108:121].strip()
                        price = int(price) / 100

                        date_string = line[2:10].strip()
                        datetime_object = datetime.strptime(
                            date_string, '%Y%m%d'
                        )

                        self.data[code][datetime_object] = price
                except:
                    continue

        for asset in self.data:
            self.data[asset] = pd.Series(
                data=self.data[asset],
                index=list(self.data[asset].keys())
            )

        self.data = pd.concat(self.data, axis=1)
        # order by date
        self.data = self.data.sort_index()

        historical_return = np.log(self.data / self.data.shift())
        historical_return = historical_return.dropna()
        
        self.historical_mean = historical_return.mean(axis=0).to_frame()
        self.historical_mean.columns = ['mu']
        self.historical_mean.transpose()
        self.historical_cov = historical_return.cov()
        
    def generate_random_portfolios(self, number_of_portfolios):
        port_returns = np.zeros(number_of_portfolios)
        port_stdevs = np.zeros(number_of_portfolios)

        for i in range(number_of_portfolios):
            weights = np.random.rand(self.number_of_assets)
            weights /= sum(weights)

            port_return = 250 * np.dot(
                weights.T, 
                self.historical_mean.to_numpy()
            )
            
            port_stdev = np.sqrt(250) * np.sqrt(
                np.dot(
                        weights.T, 
                        np.dot(self.historical_cov, weights)
                )
            )

            port_returns[i] = port_return[0]
            port_stdevs[i] = port_stdev
        
        self.random_portfolios_returns = port_returns
        self.random_portfolios_stdevs = port_stdevs

    def gmv_portfolio(self):
        # Closed form solution

        historical_cov_inv = np.linalg.inv(self.historical_cov)
        one_vector = np.ones(self.number_of_assets)
        gmv_weights = np.dot(historical_cov_inv, one_vector) / (
            np.dot(
                np.transpose(one_vector), 
                np.dot(historical_cov_inv, one_vector)
            )
        )
        gmv_weights = pd.DataFrame(data=gmv_weights).transpose()
        gmv_weights.columns = self.data.columns
        gmv_stdev = np.sqrt(250) * np.sqrt(
            np.dot(
                gmv_weights.T,
                np.dot(self.historical_cov, gmv_weights)
            )
        )

        return gmv_weights
    
    def max_return_portfolio(self):
        # Numerical solution

        one_vector = np.ones(self.number_of_assets)

        max_expected_return = np.max(self.historical_mean)
        P = matrix(self.historical_cov.to_numpy())
        q = matrix(np.zeros((self.number_of_assets, 1)))
        A = matrix(
            np.hstack([
                np.array(self.historical_mean), 
                one_vector.reshape(self.number_of_assets, 1)
            ]).transpose()
        )
        b = matrix([max_expected_return, 1])
        max_return_weights = np.array(solvers.qp(P, q, A=A, b=b)['x'])
        max_return_weights = pd.DataFrame(max_return_weights).transpose()
        max_return_weights.columns = self.data.columns

        return max_return_weights
    
    def tangency_portfolio(self):
        risk_free_interest_rate = 0.01
        one_vector = np.ones(self.number_of_assets)
        historical_cov_inv = np.linalg.inv(self.historical_cov)

        sharpe_weights = (
            np.dot(
                historical_cov_inv, 
                self.historical_mean.to_numpy() - risk_free_interest_rate / 250
            ) / 
            np.dot(
                one_vector, 
                np.dot(
                    historical_cov_inv, 
                    self.historical_mean.to_numpy() - 
                    risk_free_interest_rate / 250
                )
            )
        )

        sharpe_weights = pd.DataFrame(sharpe_weights).T
        sharpe_weights.columns = self.data.columns
        return sharpe_weights

    def generate_efficient_frontier(self):
        # number of portfolios on the efficient frontier
        N = 100
        min_return = min(self.historical_mean.to_numpy())
        max_return = max(self.historical_mean.to_numpy())
        target_returns = np.linspace(min_return, max_return, N)
        one_vector = np.ones(self.number_of_assets)

        P = matrix(self.historical_cov.to_numpy())
        q = matrix(np.zeros((self.number_of_assets, 1)))
        A = matrix(
            np.hstack([
                np.array(self.historical_mean), 
                one_vector.reshape(self.number_of_assets, 1)
            ]).transpose()
        )

        optimal_weights = []
        for target_return in target_returns:
            optimal_weights.append(
                solvers.qp(
                    P, q, A=A, b=matrix([[target_return.item(),1]])
                )['x']
            )

        efficient_frontier_returns = []
        efficient_frontier_risks = []
        for weigth in optimal_weights:
            efficient_frontier_returns.append(
                np.dot(weigth.T, self.historical_mean.to_numpy()).item()*250
            )
            efficient_frontier_risks.append(
                np.sqrt(
                    np.dot(weigth.T, np.dot(self.historical_cov, weigth)) * 250
                ).item()
            )

        return efficient_frontier_returns, efficient_frontier_risks

    def plot_random_portofolios(self):
        plt.plot(
            self.random_portfolios_stdevs,
            self.random_portfolios_returns,
            'o',
            markersize=6
        )
        plt.xlabel('expected volatility')
        plt.ylabel('expected return')
        plt.title('Random portfolios')
        plt.savefig("./results/volxreturn.png")
        plt.close()

    def plot_efficient_frontier(
        self,
        efficient_frontier_returns,
        efficient_frontier_risks
    ):
        plt.plot(
            self.random_portfolios_stdevs,
            self.random_portfolios_returns,
            'o',
            markersize=6,
            label='Candidate portfolios'
        )
        plt.plot(
            efficient_frontier_risks,
            efficient_frontier_returns,
            color='green',
            markersize=8,
            label='Efficient frontier'
        )
        plt.xlabel('expected volatility')
        plt.ylabel('expected return')
        plt.title('Efficient frontier and candidate portfolios')
        plt.legend(loc='best')
        plt.savefig("./results/volxreturn.png")
        plt.close()

    def plot_weight_transition(self, optimal_weights):
        transition_data = pd.DataFrame(optimal_weights)
        transition_data.columns = self.data.columns
        plt.stackplot(
            range(50),
            transition_data.iloc[:50,:].T,
            labels=transition_data.columns
        )
        plt.legend(loc='upper left')
        plt.margins(0,0)
        plt.title('Transition of weights')
        plt.savefig("./results/allocation_trasition_matrix.png")