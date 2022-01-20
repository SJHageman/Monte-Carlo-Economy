# Monte-Carlo-Economy
The goal of this project is to simulate wealth distribution of a closed economy.  It originally was written in c++ for a computational physics course, but has since been moved over to python.  It is purely to satisfy my own curiosity.  It follows from the work done by [Patricia, et al.](https://doi.org/10.1016/j.physa.2004.04.024) where they develop an analytical form of a non-Gibbs wealth distribution.

The idea is to use a Markov chain Monte Carlo method to generate a wealth distribution from an arbitrary inital state.  This is done by creating a set of financial agents with an initial amount of money m0 (m0 = 1 for all calculations). Then two agents are randomly chosen, and they exchange a random amount of wealth.  Money is conserved during this interaction, and interactions that would introduce a debt are not allowed.  An additional paramter can be introduce which forces each agent to save a percentage of their total wealth.  A set of agents go through with 10^6 - 10^7 transactions (this is one chain), and then a new set of agents are generated and this is repeated for 10^3-10^4 chains.  Total, this is 10^9-10^11 transactions over all chains, so computational efficiency is important.  To that end, in the script which generates the figures and data (linked [here](https://github.com/SJHageman/Monte-Carlo-Economy/blob/main/wealth_distribution_MC.py)), all the chains are computeted in parallel and this scales with the number of cores available. With a 3700x it is possible to compute 10^3 chains with 10^7 transactions over a day.

The resulting wealth distributions for saving percentages of (0, 0.25, 0.5, 0.75) are shown below
![wealth_hists](https://github.com/SJHageman/Monte-Carlo-Economy/blob/main/hist_ind_logNumTrans%3D6_runsN%3D1000.png)

A few features immediately jump out from the distributions.  First of all, as the saving percentage increases the high-wealth tail of distribution shrinks dramatically.  Additionally, the low-wealth portion of the distribution shifts to larger wealth values, and a peak forms around the initial wealth value given to each agent.

To verify that the simulation is approaching the correct distribution, the numerically simulated results can be compared to the analystical form given by [Patricia, et al.](https://doi.org/10.1016/j.physa.2004.04.024).  This is shown is following figure by looking at the probability density function for each case
![wealth_pdf](https://github.com/SJHageman/Monte-Carlo-Economy/blob/main/hist_all_logNumTrans%3D6_runsN%3D1000.png)

The agreement between the two is reasonable, and would probably improve with a longer burn-in for each chain and more chains (10^7 transactions for 10^4 chains).

The natural inclincation is to compare these distributions with real-world wealth distributions to understand wealth inequality, however great care must be taken when doing so.  This is an extremely simple system that cannot capture the incredibly complicated macro and micro economic forces and economic policies that determine how wealth is distributed within a given society.  Nonetheless, the exponential behaviour of the low savings model reasonably captures what is seen in real wealth distributions.  They can generally be described by a power-law (Pareto) distributions for the high-wealth tail.

Next steps for this project are increases the number of chains and transactions, and looking at how the number of transactions correlates with wealth.

