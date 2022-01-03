import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.special import gamma
from matplotlib.colors import LinearSegmentedColormap

def agent_ledger_MCMC(num_transactions=1e5, num_agents=500, 
                 save_percent=0.0, m0=1.0):
    agents = np.array([m0*np.ones(num_agents),np.zeros(num_agents)]).T
    rng = np.random.default_rng()
    ledger = []
    append = ledger.append
    for n in tqdm(range(int(num_transactions))):
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
            append([id0, id1, m_0_new, m_1_new])
    return np.array(ledger)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib import colors
    
    num_runs = 100
    num_transactions = 1e7
    num_agents = 500
    save_percent = 0.0

    fname = f'ledger_agentsN={num_agents}_savep={save_percent:.2f}_logNumTrans={np.log10(num_transactions):.0f}.npy'
    try:
        ledger = np.load(fname, allow_pickle=True)
    except FileNotFoundError:
        ledger = agent_ledger_MCMC(num_transactions=num_transactions, 
                                    num_agents=num_agents, save_percent=save_percent)

        np.save(fname, ledger)

    bin_size = 0.05
    hist_bins = np.arange(0,20,bin_size)

    agents = np.ones(num_agents)
    hists = []
    append = hists.append
    target_number = 10000
    ratio = int(num_transactions/target_number)
    for i,trans in enumerate(tqdm(ledger)):
        id0, id1, m_0_new, m_1_new = trans
        agents[id0] = m_0_new
        agents[id1] = m_1_new
        #if i%ratio == 0:
        if i<target_number:
            counts, bin_edges = np.histogram(agents, bins=hist_bins)
            append(counts)
    hists = np.array(hists).T

    x = np.arange(target_number)
    fig, ax = plt.subplots()
    ax.pcolormesh(x, hist_bins, hists, norm=colors.LogNorm())
    fn = 'ledger_hists.png'
    #fig.savefig(fn)

    