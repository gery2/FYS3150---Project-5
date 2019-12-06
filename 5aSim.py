#5a): Simulation of Transactions, and 5b): Recongnizing the distribution.
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.rcParams.update({'font.size': 14})

max = 7 #max money value
n = 10**6 #10**7
N = 500 #number of agents
m0 = 1
mc = 10**3 #10**3 - 10**4
dm = 7/500 #0.01 - 0.05 (7/500=0.014)
beta = 1/m0

#Transactions
@jit(nopython=True)
def trans(m, n, m_):
    for k in range(n):
        eps = np.random.uniform(0,1)
        i = np.random.randint(0,N)
        j = i
        while i == j:
            j = np.random.randint(0,N)

        m_[i] = eps*(m[i] + m[j])
        m_[j] = (1 - eps)*(m[i]+ m[j])
        m[i] = m_[i]
        m[j] = m_[j]

    return m

#Monte Carlo cycles
def MC(n,N,m0,mc,dm,beta):
    M = np.zeros(len(np.arange(0, max + dm, dm))-1) #manually deciding limits
    for l in tqdm(range(mc)):

        m = np.array([m0]*N, dtype=np.float64)
        m_ = np.zeros(N)

        m = trans(m, n, m_)

        bins = np.arange(0, max + dm, dm)
        array, bins = np.histogram(m, bins=bins)

        M += array

    plt.show()
    return M

M = MC(n,N,m0,mc,dm,beta)
M = M/np.sum(M)
x = np.linspace(0,max,len(M))

plt.figure(figsize=(10,6))
plt.plot(x, M)
plt.title('Money per agent. 10^%d transactions. MC = %d' %(int(np.log10(n)), mc))
plt.xlabel('money')
plt.ylabel('amount of agents')
plt.show()

wm_analytical = beta*np.exp(-beta*x)

plt.figure(figsize=(10,6))
plt.semilogy(x, M/dm)
plt.semilogy(x, wm_analytical)
plt.title('Logarithmic Gibbs distribution as function of money m')
plt.xlabel('money')
plt.ylabel('log(wm)')
plt.legend(('Numerical', 'Analytical'))
plt.show()



#
