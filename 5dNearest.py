#5d): Nearest neighbor interactions. Calculations, no plots.
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import special, stats
plt.rcParams.update({'font.size': 14})

max = 100 #max value for money / x-value, around 100 here.
n = 10**5 #10**7
N = 1000 #500, 1000
m0 = 1 #starter money
mc = 300 #10**3 - 10**4
dm = 0.01 #bin width
beta = 1/m0
lam = 0 #0, 0.5

#Transactions
@jit(nopython=True)
def fnunc(m, n, m_, alfa):
    for k in range(n):
        eps = np.random.uniform(0,1)
        i = np.random.randint(0,N)
        j = i
        while i == j:
            j = np.random.randint(0,N)

        dm = (1 - lam)*(eps*m[j] - (1 - eps)*m[i])

        if abs(m[i] - m[j]) < 1e-6:
            p = 1
        else:
            p = abs(m[i] - m[j])**-alfa #likelihood for interaction

        random = np.random.uniform(0,1)

        if p > random:
            m_[i] = m[i] + dm
            m_[j] = m[j] - dm
            m[i] = m_[i]
            m[j] = m_[j]

    return m

#Monte Carlo cycles
def func(n,N,m0,mc,dm,beta,alfa):
    M = np.zeros(len(np.arange(0, max + dm, dm))-1) #manually deciding limits
    for l in tqdm(range(mc)):

        m = np.array([m0]*N, dtype=np.float64)
        m_ = np.zeros(N)

        m = fnunc(m, n, m_, alfa)

        bins = np.arange(0, max + dm, dm)
        array, bins = np.histogram(m, bins=bins) #calculating without plotting.

        M += array

    return M

#Calculating transactions
M1 = func(n,N,m0,mc,dm,beta,alfa=0.5)
M1 = M1/np.sum(M1)/dm
M2 = func(n,N,m0,mc,dm,beta,alfa=4)
M2 = M2/np.sum(M2)/dm
M3 = func(n,N,m0,mc,dm,beta,alfa=10)
M3 = M3/np.sum(M3)/dm
M4 = func(n,N,m0,mc,dm,beta,alfa=50)
M4 = M4/np.sum(M4)/dm

from tempfile import TemporaryFile
outfile = TemporaryFile()
np.save('finaltest_test', (M1,M2,M3,M4)) #saving calculations for later plotting




#
