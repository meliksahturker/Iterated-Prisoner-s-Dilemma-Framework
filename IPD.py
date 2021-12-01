import pandas as pd
import numpy as np
from scipy.special import comb

from agent import Agent

S = 0
P = 1
R = 3
T = 5
class IPD:
    """
    number_of_agents (int): must be multiple of number of groups.
    For this project, must be multiple of 6.

    number_of_iterations (int): number of games played including refusals

    memory_ratio (float): in range of [0, 1]
    """
    def __init__(self, number_of_agents, memory_ratio, strat):

        self.number_of_agents = number_of_agents
        self.memory_ratio = memory_ratio
        self.strat = strat
        self.memory_size = int(number_of_agents * memory_ratio)
        self.number_of_played_games = 0
           
        number_of_agent_in_each_group = number_of_agents // 6   

        gradient_of_coop_probs = np.linspace(0, 1, number_of_agent_in_each_group)

        # to reproduce cinar2020
        #gradient_of_coop_probs = [0.9, 0.1] * 10 # 
            
        # Initialize Agents
        self.agents = []
        for index in range(number_of_agents):
            coop_prob = round(gradient_of_coop_probs[index // 6], 2)
            
            if strat == 'mixed':
                strategy = index % 6
            else:
                strategy = strat
            memory_size = self.memory_size

            self.agents.append(Agent(index, coop_prob, strategy, memory_size, number_of_agents))

        # Initialize df_report table
        strategies = np.unique([agent.strategy for agent in self.agents])
        self.df_report = pd.DataFrame(np.zeros((len(strategies), 2)),index = strategies, columns = ['P_C', 'P_D'], dtype = int)

    def run_simulation(self, number_of_iterations):
        """
        number_of_iterations (int): number of matches
        """
        for _ in range(number_of_iterations):

            number_of_agents = self.number_of_agents

            # Select two agents randomly
            first_agent = np.random.randint(0, number_of_agents)
            second_agent = np.random.randint(0, number_of_agents)

            while first_agent == second_agent:
                second_agent = np.random.randint(0, number_of_agents)
            
            # PLAY
            they_play = self.play_single_game(self.agents[first_agent], self.agents[second_agent])
            self.number_of_played_games += they_play
    
    def play_single_game(self, agent_one: Agent, agent_two: Agent):
        first_agent: int = agent_one.index
        second_agent: int = agent_two.index

        # Will they play?
        # If first agent has a memory
        if agent_one.memory_size > 0:
            memory_of_first = agent_one.return_memory()

            # If first agent does NOT know the second agent, or knows it as cooperator, first_agent_plays = True
            if ((second_agent in memory_of_first.index.values) == False) or Agent.perceived_coop_ratio(memory_of_first.loc[second_agent, 'c'], memory_of_first.loc[second_agent, 'd']) > 0.5:
                first_agent_plays = True
            else:
                first_agent_plays = False
        # In absence of memory, they always play
        else:
            first_agent_plays = True
        
        # If second agent has a memory
        if agent_two.memory_size > 0:
            # If second agent does NOT know the first agent, or knows it as cooperator, second_agent_plays = True
            memory_of_second = agent_two.return_memory()
            if ((first_agent in memory_of_second.index.values) == False) or Agent.perceived_coop_ratio(memory_of_second.loc[first_agent, 'c'], memory_of_second.loc[first_agent, 'd']) > 0.5:
                second_agent_plays = True
            else:
                second_agent_plays = False
                
        # In absence of memory, they always play
        else:
            second_agent_plays = True
                
        they_play = first_agent_plays & second_agent_plays

        # If they play
        if they_play:
            # Check and open room in memory:
            # First agent:
            if (agent_one.memory_size > 0) and (memory_of_first.shape[0] >= agent_one.memory_size) and ((second_agent in memory_of_first.index.values) == False):
                agent_one.open_space_in_memory()
            
            # Second agent
            if (agent_two.memory_size > 0) and (memory_of_second.shape[0] >= agent_two.memory_size) and ((first_agent in memory_of_second.index.values) == False):
                agent_two.open_space_in_memory()

            # Decide whether they will coop or defect depending on their character
            # first_agent
            rand_number = np.random.uniform(low = 0, high = 1)
            if rand_number < agent_one.coop_prob:
                first_will_coop = True
            else:
                first_will_coop = False

            # second_agent
            rand_number = np.random.uniform(low = 0, high = 1)
            if rand_number < agent_two.coop_prob:
                second_will_coop = True
            else:
                second_will_coop = False

            # Update their payoff and memory
            if (first_will_coop == True) and (second_will_coop == True):
                agent_one.payoff += R
                agent_two.payoff += R

                if (agent_one.memory_size > 0):
                    agent_one.df_memory.loc[second_agent, 'c'] += 1
                    agent_one.update_pcr(second_agent)
                if (agent_two.memory_size > 0):    
                    agent_two.df_memory.loc[first_agent, 'c'] += 1
                    agent_two.update_pcr(first_agent)

            elif (first_will_coop == True) and (second_will_coop == False):
                agent_one.payoff += S
                agent_two.payoff += T

                if (agent_one.memory_size > 0):
                    agent_one.df_memory.loc[second_agent, 'd'] += 1
                    agent_one.update_pcr(second_agent)
                if (agent_two.memory_size > 0):    
                    agent_two.df_memory.loc[first_agent, 'c'] += 1
                    agent_two.update_pcr(first_agent)

            elif (first_will_coop == False) and (second_will_coop == True):
                agent_one.payoff += T
                agent_two.payoff += S

                if (agent_one.memory_size > 0):
                    agent_one.df_memory.loc[second_agent, 'c'] += 1
                    agent_one.update_pcr(second_agent)
                if (agent_two.memory_size > 0):    
                    agent_two.df_memory.loc[first_agent, 'd'] += 1
                    agent_two.update_pcr(first_agent)

            else:
                agent_one.payoff += P
                agent_two.payoff += P

                if (agent_one.memory_size > 0):
                    agent_one.df_memory.loc[second_agent, 'd'] += 1
                    agent_one.update_pcr(second_agent)
                if (agent_two.memory_size > 0):    
                    agent_two.df_memory.loc[first_agent, 'd'] += 1
                    agent_two.update_pcr(first_agent)

        return they_play

    def report_simulation_results(self):    
        num_cooperators = 0
        num_defectors = 0
        for agent in self.agents:
            # If agent is Cooperator
            if agent.coop_prob > 0.5:
                num_cooperators += 1
                self.df_report.loc[agent.strategy, 'P_C'] += agent.payoff
            # If agent is Defector
            else:
                num_defectors += 1
                self.df_report.loc[agent.strategy, 'P_D'] += agent.payoff

        strategies = np.unique([agent.strategy for agent in self.agents])
        
        num_cooperators_in_each_strat = int(num_cooperators / len(strategies))
        num_defectors_in_each_strat = int(num_defectors / len(strategies))
        
        # calculating \phi
        for strat in strategies: # for each strategy
            numerator = self.df_report.loc[strat, 'P_C'] / num_cooperators_in_each_strat
            sum_of_all = self.df_report.sum().sum()
            denominator = sum_of_all / (num_cooperators + num_defectors)
            self.df_report.loc[strat, 'phi_c'] = numerator / denominator

        return self.df_report
        
    def report_raw_payoff_table(self):
        df = pd.DataFrame()
        if len(np.unique([agent.strategy for agent in self.agents])) > 1:
            env = 'mixed'
        else:
            env = 'not_mixed'
            
        for idx, agent in enumerate(self.agents):

            df.loc[idx, 'env'] = env
            df.loc[idx, 'strat'] = agent.strategy
            df.loc[idx, 'coop_prob'] = agent.coop_prob
            df.loc[idx, 'memory_size'] = agent.memory_size
            df.loc[idx, 'payoff'] = agent.payoff

        int_cols = ['strat', 'memory_size', 'payoff']
        for col in int_cols:
            df.loc[:, col] = df.loc[:, col].astype(int)
        
        return df


def run_single_experiment(args):
    number_of_agents, tau, memory_ratio, strat = args
    number_of_iterations = int(comb(number_of_agents, 2)) * tau
    
    ipd = IPD(number_of_agents, memory_ratio, strat)
    ipd.run_simulation(number_of_iterations)
    
    df_report = ipd.report_simulation_results()
    df_raw_payoffs = ipd.report_raw_payoff_table()

    number_of_played_games = ipd.number_of_played_games
    if ipd.strat == 'mixed':
        phi_X_results = df_report.loc[:, 'phi_c'].values.tolist()
    else:
        _lst_of_nan_c = [np.nan for _ in range(6)]
        _lst_of_nan_c[ipd.strat] = df_report.loc[:, 'phi_c'].values.tolist()[0]
        
        phi_X_results = _lst_of_nan_c

    result = [strat, memory_ratio, number_of_played_games] + phi_X_results
    
    return result, df_raw_payoffs