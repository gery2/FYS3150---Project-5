#5e): Nearest neighbors and former transactions. General analysis plot + degrees plot included.
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import special
plt.rcParams.update({'font.size': 14})

max = 100 #max value for money / x-value
n = 10**6 #1024 for the degree plot, 10**7 otherwise
N = 1000 #1000
m0 = 1 #starter money
mc = 1000 #10**3 - 10**4
dm = 0.01 #bins width
beta = 1/m0
lam = 0 #0, 0.25

#Transactions
@jit(nopython=True)
def fnunc(m, n, m_, alfa, gamma):
    matrix = np.zeros((N,N))
    degree = np.zeros(N)

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
            p = abs(m[i] - m[j])**-alfa*(matrix[i][j] + 1)**gamma #likelihood for interaction

        random = np.random.uniform(0,1)

        if p > random:
            m_[i] = m[i] + dm
            m_[j] = m[j] - dm
            m[i] = m_[i]
            m[j] = m_[j]
            matrix[i][j] += 1 #counting number of transactions between i, j
            matrix[j][i] += 1 #also counting j, i because it's the same interaction
            degree[i] += 1.
            degree[j] += 1.


    return m, degree

#Monte Carlo cycles
def func(n,N,m0,mc,dm,beta,alfa, gamma):
    M = np.zeros(len(np.arange(0, max + dm, dm))-1) #manually deciding limits
    Degree = np.zeros(21)
    for l in tqdm(range(mc)):

        m = np.array([m0]*N, dtype=np.float64)
        m_ = np.zeros(N)

        m, degree = fnunc(m, n, m_, alfa, gamma)
        #print(degree)
        bins = np.arange(0, max + dm, dm)
        array, bins = np.histogram(m, bins=bins) #calculating without plotting

        binses = np.linspace(0,20,22)
        array2, bins2 = np.histogram(degree, bins=binses) #calculating without plotting.

        M += array
        Degree += array2
    return M, Degree

M10, Degree10 = func(n,N,m0,mc,dm,beta,alfa=1,gamma=0)
M10 = M10/np.sum(M10)/dm; Degree10 = Degree10 / mc / np.sum(Degree10 / mc)
M11, Degree11 = func(n,N,m0,mc,dm,beta,alfa=1,gamma=1)
M11 = M11/np.sum(M11)/dm; Degree11 = Degree11 / mc / np.sum(Degree11 / mc)
M12, Degree12 = func(n,N,m0,mc,dm,beta,alfa=1,gamma=2)
M12 = M12/np.sum(M12)/dm; Degree12 = Degree12 / mc / np.sum(Degree12 / mc)
M13, Degree13 = func(n,N,m0,mc,dm,beta,alfa=1,gamma=3)
M13 = M13/np.sum(M13)/dm; Degree13 = Degree13 / mc / np.sum(Degree13 / mc)
M14, Degree14 = func(n,N,m0,mc,dm,beta,alfa=1,gamma=4)
M14 = M14/np.sum(M14)/dm; Degree14 = Degree14 / mc / np.sum(Degree14 / mc)

M20, Degree20 = func(n,N,m0,mc,dm,beta,alfa=2,gamma=0)
M20 = M20/np.sum(M20)/dm; Degree20 = Degree20 / mc / np.sum(Degree20 / mc)
M21, Degree21 = func(n,N,m0,mc,dm,beta,alfa=2,gamma=1)
M21 = M21/np.sum(M21)/dm; Degree21 = Degree21 / mc / np.sum(Degree21 / mc)
M22, Degree22 = func(n,N,m0,mc,dm,beta,alfa=2,gamma=2)
M22 = M22/np.sum(M22)/dm; Degree22 = Degree22 / mc / np.sum(Degree22 / mc)
M23, Degree23 = func(n,N,m0,mc,dm,beta,alfa=2,gamma=3)
M23 = M23/np.sum(M23)/dm; Degree23 = Degree23 / mc / np.sum(Degree23 / mc)
M24, Degree24 = func(n,N,m0,mc,dm,beta,alfa=2,gamma=4)
M24 = M24/np.sum(M24)/dm; Degree24 = Degree24 / mc / np.sum(Degree24 / mc)

from tempfile import TemporaryFile
outfile = TemporaryFile()
np.save('outfile2', (M10,M11,M12,M13,M14,M20,M21,M22,M23,M24)) #saving calculations for later plotting

x = np.linspace(0,max,N)
x2 = np.linspace(0,max,len(np.arange(0, max + dm, dm))-1)
k = np.linspace(0,20,21)

#plotting amount of agents per money-amount. alpha = 1
#use n = 10**7, N = 1000
plt.figure(figsize=(10,6))
plt.loglog(x2,M10)
plt.loglog(x2,M11)
plt.loglog(x2,M12)
plt.loglog(x2,M13)
plt.loglog(x2,M14)
plt.title('Money per agent. 10^%d transactions, MC = %d, α = 1.0 ' %(int(np.log10(n)), mc))
plt.xlabel('Money')
plt.ylabel('amount of agents')
plt.legend(('γ = 0.0', 'γ = 1.0', 'γ = 2.0', 'γ = 3.0', 'γ = 4.0'))
plt.xlim(0.1,max/2)
plt.show()

#plotting amount of agents per money-amount. alpha = 2
#use n = 10**7, N = 1000
plt.figure(figsize=(10,6))
plt.loglog(x2,M20)
plt.loglog(x2,M21)
plt.loglog(x2,M22)
plt.loglog(x2,M23)
plt.loglog(x2,M24)
plt.title('Money per agent. 10^%d transactions, MC = %d, α = 2.0 ' %(int(np.log10(n)), mc))
plt.xlabel('Money')
plt.ylabel('amount of agents')
plt.legend(('γ = 0.0', 'γ = 1.0', 'γ = 2.0', 'γ = 3.0', 'γ = 4.0'))
plt.xlim(0.1,max)
plt.show()

#plotting degree distribution D(k) as function of degree k. alfa = 1
#use n = 1024, N = 1024 (like in FIG.6)
plt.figure(figsize=(10,6))
plt.semilogy(k,Degree10)
plt.semilogy(k,Degree11)
plt.semilogy(k,Degree12)
plt.semilogy(k,Degree13)
plt.semilogy(k,Degree14)
plt.title('Degree distribution. alpha = 1')
plt.xlabel('k')
plt.ylabel('D(k)')
plt.legend(('γ = 0.0', 'γ = 1.0', 'γ = 2.0', 'γ = 3.0', 'γ = 4.0'))
plt.show()

#plotting degree distribution D(k) as function of degree k. alfa = 2
#use n = 1024, N = 1024 (like in FIG.6)
plt.figure(figsize=(10,6))
plt.semilogy(k,Degree20)
plt.semilogy(k,Degree21)
plt.semilogy(k,Degree22)
plt.semilogy(k,Degree23)
plt.semilogy(k,Degree24)
plt.title('Degree distribution. alpha = 2')
plt.xlabel('k')
plt.ylabel('D(k)')
plt.legend(('γ = 0.0', 'γ = 1.0', 'γ = 2.0', 'γ = 3.0', 'γ = 4.0'))
plt.show()




#
