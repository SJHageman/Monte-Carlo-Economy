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
//		initialize population of N=500 agents with m_0=1 => <m>=1
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

import numpy as np
from tqdm import tqdm
from scipy.special import gamma
from matplotlib.colors import LinearSegmentedColormap

def agent_MCMC(num_transactions=1e5, num_agents=500, 
                 save_percent=0.0, m0=1.0):
    """
    Calculate Markov chain for a set of financial agents by allowing for
    random transactions beteween them.

    Parameters
    ----------
    num_transactions : int, optional
        Number of transactions. The default is 1e5.
    num_agents : int, optional
        Number of agents. The default is 500.
    save_percent : float, optional
        Saving percentage for each agent. The default is 0.0.
    m0 : float, optional
        Inital amount of money each agent is given. The default is 1.0.

    Returns
    -------
    agents : numpy array
        Array of agents wealth and number of transactions.

    """
    agents = np.array([m0*np.ones(num_agents),np.zeros(num_agents)]).T
    rng = np.random.default_rng()
    for n in range(int(num_transactions)):
        id0,id1 = rng.integers(0, high=num_agents, size=2)
        eps = rng.random(1)
        m_0 = agents[id0,0]
        m_1 = agents[id1,0]
        delta_m = (1.-save_percent) * (eps * m_1 - (1.-eps) * m_0)
        m_0_new = m_0 + delta_m
        m_1_new = m_1 - delta_m
        if m_0_new>0 and m_1_new>0:
            agents[id0,0] = m_0_new
            agents[id1,0] = m_1_new
            agents[id0,1] += 1
            agents[id1,1] += 1
    return agents

def extended_Gibbs_dist(m, lam):
    """
    Probability density describing non-Gibbs dynamics.

    Parameters
    ----------
    m : array
        Wealth of agents.
    lam : float
        Saving percent.

    Returns
    -------
    p_n : array
        Probability density.

    """
    n = 1 + 3*lam/(1-lam)
    a_n = n**n/gamma(n)
    p_n = a_n*m**(n-1)*np.exp(-n*m)
    return p_n

def cmapy(num_of_lines):
    """
    Generate list of colors for plotting.

    Parameters
    ----------
    num_of_lines : int
        Number of lines.

    Returns
    -------
    colorsy : list
        List of colors.

    """
    start = 0.0 
    stop = 1.0 
    number_of_lines = num_of_lines
    cm_subsection = np.linspace(start, stop, number_of_lines) 
    cmapy_RB = LinearSegmentedColormap.from_list('mycmap', ['darkred', 'purple', 'darkblue','dodgerblue'])
    colorsy = [ cmapy_RB(x) for x in cm_subsection ]
    return colorsy

if __name__ == '__main__':
    from joblib import Parallel, delayed
    import matplotlib.pyplot as plt
    import contextlib
    import joblib

    # hyperparameters for MCMC run
    num_runs = 1000
    n_jobs = -2
    num_transactions = 1e6
    num_agents = 500
    save_percent_list = [0.0, 0.25, 0.5, 0.75]

    # booleans to determine whether to save generated agent arrays and the descriptive figures
    save_arrays = True
    save_figure = True

    #change joblibs progress bar to the tqdm progress bar
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

    #initialize figures
    fig_pdf, ax = plt.subplots()
    fig_hist, axs = plt.subplots(4,1, sharex=True, sharey=True, figsize=(6,8))
    fig_corr, ax_corr = plt.subplots()
    colorsy = cmapy(len(save_percent_list))#plt.cm.jet(np.linspace(0,1,len(save_percent_list)))

    # initialize bins for histograms
    bin_size = 0.05
    hist_bins = np.arange(0,20,bin_size)

    # run MCMC for different saving parameters
    for z,a,c,save_percent in zip(np.arange(len(save_percent_list))[::-1], axs.flatten(), colorsy, save_percent_list):
        # filename of saved npy array
        fname = f'agentsN={num_agents}_savep={save_percent:.2f}_logNumTrans={np.log10(num_transactions):.0f}_runsN={num_runs}_transN.npy'

        # try and load array, calculate if not found
        try:
            agent_runs = np.load(fname, allow_pickle=True)
        except FileNotFoundError:
            with tqdm_joblib(tqdm(desc=f"Save percent {save_percent:.2f}", total=num_runs)) as progress_bar:
                agent_runs = Parallel(n_jobs=n_jobs)(delayed(agent_MCMC)
                                (num_transactions=num_transactions, 
                                num_agents=num_agents, save_percent=save_percent) 
                                for n in range(num_runs))
                agent_runs = np.array(agent_runs)
            if save_arrays:
                np.save(fname, agent_runs)

        # get agent wealth and number of transactions
        agent_wealth = agent_runs[:,:,0]
        agent_transactions = agent_runs[:,:,1]

        # generate and plot histograms
        count, bins, _ = a.hist(agent_wealth.flatten(), bins=hist_bins, 
                        label=save_percent, color=c, zorder=z)
        a.legend()
        a.set_yscale('log')
        a.set_xlim(0,12)
        count, bins, _ = ax.hist(agent_wealth.flatten(), bins=hist_bins, alpha=0.0,
                                    color=c, density=True, stacked=True, zorder=z)
        bin_centers = bins[:-1] + 0.5*bin_size
        m_avg = np.sum(agent_runs[:,:,0])/(num_runs*num_agents)
        ax.plot(bin_centers, count, 'o', c=c, mec='black', label=save_percent)
        ax.plot(bin_centers, extended_Gibbs_dist(bin_centers/m_avg,save_percent), c=c)

        #plot correlation coefficient between number of transactions and wealth
        correlation_coef = np.corrcoef(agent_wealth.flatten(), agent_transactions.flatten())[0,1]
        ax_corr.plot(save_percent, correlation_coef, 'o', c=c, alpha=0.8, mec='black')

    # set final figure params
    ax.set_xlim(0,2.5)
    ax.legend()
    fig_pdf.supylabel('Probability density')
    fig_pdf.supxlabel('Wealth')
    fig_hist.supxlabel('Wealth')
    fig_hist.supylabel('Counts')
    fig_hist.tight_layout()
    fig_pdf.tight_layout()

    # save figures
    if save_figure:
        fig_pdf.savefig(f'hist_all_logNumTrans={np.log10(num_transactions):.0f}_runsN={num_runs}.png')
        fig_hist.savefig(f'hist_ind_logNumTrans={np.log10(num_transactions):.0f}_runsN={num_runs}.png')
        fig_corr.savefig('slope_vs_savings.png')
    
    
    

