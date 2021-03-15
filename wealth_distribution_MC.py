"""
//**************************************************************************
//	file: wealth_dist.cpp
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
import matplotlib.pyplot as plt

num_agents = 500
num_transactions = 2

df = pd.DataFrame({'agents':np.ones(num_agents)})

for n in range(num_transactions):
    transaction_df = df.sample(n=2)
    print(transaction_df[0])

