"""
//**************************************************************************
//	file: wealth_distribution_MC.py
//
//	Program to calculate the wealth distribution in a closed economy.
//
//	Programmer: Stephen J. Hageman
//
//	Revision history:
//		March 15, 2021 original version
//
//
//	Notes:
//
//
//
//**************************************************************************
//	Pseudocode:
//		initialize population of N=500 agents with m_0=500 => <m>=1
//		for 10^3-10^4 runs
//			for 10^7 transactions
//				randomly pick two agents
//				trade:
//					m_i' = e(m_i+m_j)
//					m_j'= (1-e)(m_i+m_j)
//				reject if m_i' or m_j' < 0 (no debt)
//
//**************************************************************************
"""

import pandas as pd
import numpy as np

def agent_mc_run(run_number = 0, num_transactions=1e5, num_agents=500, 
                 save_percent=0.0):
    df = pd.DataFrame({'ID':np.arange(num_agents, dtype=int), 
                       'wealth':np.ones(num_agents),
                       'run_number':run_number*np.ones(num_agents)})
    for n in np.arange(num_transactions):
        transaction_df = df.sample(n=2)
        m_0 = transaction_df.iloc[0].wealth
        m_1 = transaction_df.iloc[1].wealth
        ID0 = transaction_df.iloc[0].ID
        ID1 = transaction_df.iloc[1].ID
        eps = np.random.rand(1)
        delta_m = (1.-save_percent) * (eps * m_1 - (1.-eps) * m_0)
        m_0_new = m_0 + delta_m
        m_1_new = m_1 - delta_m
        random0 = np.random.rand(1) 
        random1 = np.random.rand(1) 
        if m_0_new>0 and m_1_new>0:
            if random0<np.exp(-delta_m/m_0) and random1<np.exp(-delta_m/m_1):
                df.at[int(ID0), 'wealth'] = m_0_new
                df.at[int(ID1), 'wealth'] = m_1_new
    return df


if __name__ == '__main__':
    from joblib import Parallel, delayed
    import matplotlib.pyplot as plt
    
    num_runs = 16
    num_transactions = 1e5
    num_agents = 500
    df_list = Parallel(n_jobs=-1, verbose=10)(delayed(agent_mc_run)
                    (run_number=n, num_transactions=num_transactions, 
                     num_agents=num_agents) 
                    for n in range(num_runs))
    df_all_runs = pd.concat(df_list)
    
    df_all_runs.hist(column='wealth')
    df_all_runs.to_csv('df_all_runs.csv', index=False)
    
    
    

