import pandas as pd
import numpy as np

class Agent:
    def __init__(self, index, coop_prob, strategy, memory_size, number_of_agents):
        self.coop_prob = coop_prob
        self.payoff = 0
        self.index = index
        self.strategy = strategy
        self.memory_size = memory_size
        self.strategy = strategy

        # init memory
        self.df_memory = pd.DataFrame(np.zeros((number_of_agents, 3)), columns = ['c', 'd', 'pcr'], dtype=int)
        self.df_memory.drop(index, axis = 0, inplace = True) # drop $i$ itself
        self.df_memory.loc[:, 'pcr'] = self.df_memory.loc[:, 'pcr'].astype(float)
    
    
    # Strategies
    # 0: Forgets Randomly --- FR
    # 1: Forgets Cooperators First --- FCF
    # 2: Forgets Defectors First --- FDF
    # 3: Forgets Closest to 0.5 first --- FibF
    # 4: Forget minority regardless of C/D --- FMinF
    # 5: Forget majority regardless of C/D --- FMajF
    
    def open_space_in_memory(self):
        # sampling via frac ensures that among equal values, random one is selected
        # otherwise there would be a bias towards lower index agents, whose coop_prob is also low
        # because they are sorted due to gradient_of_coop_probs

        if self.strategy == 0:
            agent_to_forget = self.return_memory().sample(1).index.values[0]
            
        elif self.strategy == 1:
            agent_to_forget = self.return_memory().loc[:, 'pcr'].sample(frac = 1).idxmax()

            # To reproduce cinar2020
            """
            if self.return_memory().loc[self.return_memory()['pcr'] > 0.5].shape[0]> 0:
                agent_to_forget = self.return_memory().loc[self.return_memory()['pcr'] > 0.5].sample(1).index.values[0]
            else:
                agent_to_forget = self.return_memory().sample(1).index.values[0]
            """
            
        elif self.strategy == 2:
            agent_to_forget = self.return_memory().loc[:, 'pcr'].sample(frac = 1).idxmin()
            
            # To reproduce cinar2020
            """
            if self.return_memory().loc[self.return_memory()['pcr'] <= 0.5].shape[0] > 0:
                agent_to_forget = self.return_memory().loc[self.return_memory()['pcr'] <= 0.5].sample(1).index.values[0]
            else:
                agent_to_forget = self.return_memory().sample(1).index.values[0]
            """

        elif self.strategy == 3:
            agent_to_forget = self.return_memory().loc[:, 'pcr'].sub(0.5).abs().sample(frac = 1).idxmin()

        elif self.strategy == 4:
            agent_to_forget = self.return_memory().sum(axis = 1).sample(frac = 1).idxmin()
            
        elif self.strategy == 5:
            agent_to_forget = self.return_memory().sum(axis = 1).sample(frac = 1).idxmax()

        # pcr = 0.5 | c = 0, d = 0 since (0+1) / ((1+0) + (1+0)) = 0.5
        # but in fact this does not matter since agent has to play against unknown opponent
        self.df_memory.loc[agent_to_forget, :] = [0, 0, 0] 
        
    def return_memory(self):
        return self.df_memory[self.df_memory.sum(axis = 1) > 0]

    @staticmethod
    def perceived_coop_ratio(c, d):
        cinar2020 = True
        return (c+ cinar2020) / ((c+cinar2020) + (d+cinar2020))

    def update_pcr(self, j):
        c = self.df_memory.loc[j, 'c']
        d = self.df_memory.loc[j, 'd']
        self.df_memory.loc[j, 'pcr'] = self.perceived_coop_ratio(c,d)