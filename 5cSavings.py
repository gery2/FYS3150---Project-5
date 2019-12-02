#5c): Transactions and savings.
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import special, stats
plt.rcParams.update({'font.size': 14})

#max value [7, 4.5, 3.5, 2] for parametrization and power laws (for diff lam)
max = 2 #max value for money / x-value
n = 10**6 #10**7
N = 500
m0 = 1
mc = 2000 #10**3 - 10**4
dm = 0.01
beta = 1/m0
lam = 0.9 #0.25, 0.5, 0.9
n_lam = 1 + (3*lam/(1 - lam))
an = 1/(special.gamma(n_lam))*(n_lam/m0)**n_lam
fn = np.zeros(N)

#Transactions
@jit(nopython=True)
def fnunc(m, n, m_):
    for k in range(n):
        eps = np.random.uniform(0,1)
        i = np.random.randint(0,N)
        j = i
        while i == j:
            j = np.random.randint(0,N)

        dm = (1 - lam)*(eps*m[j] - (1 - eps)*m[i]) #including savings (lam)
        m_[i] = m[i] + dm
        m_[j] = m[j] - dm
        m[i] = m_[i]
        m[j] = m_[j]

    return m

#Monte Carlo cycles
def func(n,N,m0,mc,dm,beta):
    M = np.zeros(len(np.arange(0, max + dm, dm))-1) #manually deciding limits
    for l in tqdm(range(mc)):

        m = np.array([m0]*N, dtype=np.float64)
        m_ = np.zeros(N)

        m = fnunc(m, n, m_)

        bins = np.arange(0, max + dm, dm)
        array, bins = np.histogram(m, bins=bins) #calculating without plotting.

        M += array

    return M

M = func(n,N,m0,mc,dm,beta)
M = M/np.sum(M)/dm #dividing by the bins length for proper scaling
x = np.linspace(0,max,N)
for i in range(N):
    fn[i] = an*x[i]**(n_lam-1)*np.exp(-n_lam*x[i]/m0)



x2 = np.linspace(0,max,len(np.arange(0, max + dm, dm))-1)

mask = np.where(M != 0) # only looking at the values where agents have money
M_tail = M[mask][-40:] #-100: for lamda 0.25 and 0.5
x_tail = x2[mask][-40:] #-100: for lamda 0.25 and 0.5


slope, inter = stats.linregress(np.log(x_tail), np.log(M_tail))[:2]
print(slope, inter)

plt.figure(figsize=(10,6))
plt.plot(x2,M)
plt.title('Money per agent. 10^%d transactions. MC = %d' %(int(np.log10(n)), mc))
plt.xlabel('Money')
plt.ylabel('amount of agents')
plt.plot(x, fn)
plt.legend(('Numerical', 'Analytical'))
plt.show()

plt.figure(figsize=(10,6))
plt.loglog(x2,M)
plt.title('Money per agent. 10^%d transactions. MC = %d' %(int(np.log10(n)), mc))
plt.xlabel('Money')
plt.ylabel('amount of agents')
plt.loglog(x, fn)
plt.loglog(x, (np.exp(inter)*x**(slope)))
plt.legend(('Numerical', 'Analytical', 'Power law'))
plt.xlim(x_tail[0],x_tail[-1])
plt.ylim(M_tail[0],M_tail[-1])
plt.show()


#
