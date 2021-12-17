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
from tqdm import tqdm

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
            e0 = -delta_m/m_0
            e1 = -delta_m/m_1
            if e0>709 or e1>709 or e0<-709 or e1<-709:
                pass
            else:
                if random0<np.exp(e0) and random1<np.exp(e1):
                    df.at[int(ID0), 'wealth'] = m_0_new
                    df.at[int(ID1), 'wealth'] = m_1_new
    return df

def agent_MCMC(num_transactions=1e5, num_agents=500, 
                 save_percent=0.0, m0=1.0):
    agents = m0*np.ones(num_agents)
    rng = np.random.default_rng()
    #indx_pairs = [rng.integers(0, high=num_agents, size=2) for n in range(num_transactions)]
    for n in range(int(num_transactions)):
        id0,id1 = rng.integers(0, high=num_agents, size=2)
        eps, rnd0, rnd1 = rng.random(3)
        m_0 = agents[id0]
        m_1 = agents[id1]
        delta_m = (1.-save_percent) * (eps * m_1 - (1.-eps) * m_0)
        m_0_new = m_0 + delta_m
        m_1_new = m_1 - delta_m
        if m_0_new>0 and m_1_new>0:
            e0 = -delta_m/m_0
            e1 = -delta_m/m_1
            if np.log(rnd0)<e0 and np.log(rnd1)<e1:
                agents[id0] = m_0_new
                agents[id1] = m_1_new
    return agents

if __name__ == '__main__':
    from joblib import Parallel, delayed
    import matplotlib.pyplot as plt
    import contextlib
    import joblib

    num_runs = 1000
    num_transactions = 1e7
    num_agents = 500
    save_percent_list = [0.0, 0.25, 0.5, 0.75]
    """
    for save_percent in save_percent_list:
        df_list = Parallel(n_jobs=-1, verbose=10)(delayed(agent_mc_run)
                        (run_number=n, num_transactions=num_transactions, 
                         num_agents=num_agents, save_percent=save_percent) 
                        for n in range(num_runs))
        df_all_runs = pd.concat(df_list)
        df_all_runs.to_csv(f'df_all_runs_save_{save_percent:.2f}.csv', 
                           index=False)
        
        
        fig, ax = plt.subplots()
        df_all_runs.hist(column='wealth', ax=ax)
        ax.set_yscale('log')
        #ax.set_xscale('log')
        save_figure = False
        if save_figure:
            fig.savefig(f'hist_all_runs_{save_percent:.2f}_trans_{np.log10(num_transactions):.0f}.png')
    """
    #_ = agent_MCMC(num_transactions=num_transactions, num_agents=num_agents, save_percent=save_percent_list[0])


    @contextlib.contextmanager
    def tqdm_joblib(tqdm_object):
        """Context manager to patch joblib to report into tqdm progress bar given as argument"""
        class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
            def __call__(self, *args, **kwargs):
                tqdm_object.update(n=self.batch_size)
                return super().__call__(*args, **kwargs)
        
        old_batch_callback = joblib.parallel.BatchCompletionCallBack
        joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
        try:
            yield tqdm_object
        finally:
            joblib.parallel.BatchCompletionCallBack = old_batch_callback
            tqdm_object.close()    

    save_arrays = True
    fig, ax = plt.subplots()
    colorsy = plt.cm.jet(np.linspace(0,1,len(save_percent_list)))
    fig2, axs = plt.subplots(2,2)
    for a,c,save_percent in zip(axs.flatten(),colorsy, save_percent_list):
        fname = f'agentsN={num_agents}_savep={save_percent:.2f}_logNumTrans={np.log10(num_transactions):.0f}_runsN={num_runs}.npy'
        try:
            agent_runs = np.load(fname)
        except FileNotFoundError:
            with tqdm_joblib(tqdm(desc=f"Save percent {save_percent:.2f}", total=num_runs)) as progress_bar:
                agent_runs = Parallel(n_jobs=-2)(delayed(agent_MCMC)
                                (num_transactions=num_transactions, 
                                num_agents=num_agents, save_percent=save_percent) 
                                for n in range(num_runs))          
            if save_arrays:
                np.save(fname, agent_runs)
        
        ax.hist(np.array(agent_runs).flatten(), bins=100, range=(0,100), 
                        label=save_percent, alpha=0.5, edgecolor='black', linewidth=1, color=c)
        a.hist(np.array(agent_runs).flatten(), bins=100,
                        label=save_percent, alpha=0.5, edgecolor='black', linewidth=1, color=c)
        a.set_yscale('log')
        a.legend()
    ax.set_yscale('log')
    #ax.set_xscale('log')
    ax.legend()
    save_figure = True
    if save_figure:
        fig.savefig(f'hist_all_logNumTrans={np.log10(num_transactions):.0f}_runsN={num_runs}.png')
        fig2.savefig(f'hist_ind_logNumTrans={np.log10(num_transactions):.0f}_runsN={num_runs}.png')
    
    
    

