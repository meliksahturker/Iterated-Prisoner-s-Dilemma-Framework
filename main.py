import itertools
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import numpy as np

from IPD import run_single_experiment


number_of_realization = 5
tau = 30

number_of_agents = [126]
tau = [tau] # C(126, 2) * 30 = 236_250
memory_ratio = np.arange(0, 1.05, 0.05).tolist()
strat = [0, 1, 2, 3, 4, 5, 'mixed']

list_args = number_of_realization * list(itertools.product(number_of_agents, tau, memory_ratio, strat))

accumulated_results = []
df_raw_payoff = pd.DataFrame()
with ProcessPoolExecutor() as executor:
    for returned_object in executor.map(run_single_experiment, list_args):
        result, _raw_payoff_table = returned_object
        accumulated_results.append(result)
        df_raw_payoff = pd.concat([df_raw_payoff, _raw_payoff_table])


df = pd.DataFrame(accumulated_results, columns = ['strat', 'memory_ratio', 'number_of_played_games', 'phi_c_0', 'phi_c_1', 'phi_c_2', 'phi_c_3', 'phi_c_4', 'phi_c_5'])

df.to_csv('results.csv')
df_raw_payoff.to_csv('raw_payoffs.csv')