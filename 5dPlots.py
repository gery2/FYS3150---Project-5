#plots for 5d):
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import special, stats
plt.rcParams.update({'font.size': 14})

max = 500 #max value for money / x-value
n = 10**5 #10**7
N = 500 #500, 1000
m0 = 1 #starter money
mc = 1000 #10**3 - 10**4
dm = 0.01 #bin width
beta = 1/m0
lam = 0 #0, 0.25

#lam = 0: 'outfile.npy'
(M1,M2,M3,M4) = np.load('outfile_lam.npy') #loading calculations. with lam=0.25 is outfile_lam.npy

x = np.linspace(0,max,N)
x2 = np.linspace(0,max,len(M1))
#masks for different alphas
mask1 = np.where(M1 != 0)
# only looking at the values where agents have money
i1 = 600; i2 = 900; i3 = 1900; i4 = 3800 #intervals to look at for tails.
M_tail1 = M1[mask1][-i1:]
x_tail1 = x2[mask1][-i1:]
mask2 = np.where(M2 != 0)
M_tail2 = M2[mask2][-i2:]
x_tail2 = x2[mask2][-i2:]
mask3 = np.where(M3 != 0)
M_tail3 = M3[mask3][-i3:]
x_tail3 = x2[mask3][-i3:]
mask4 = np.where(M4 != 0)
M_tail4 = M4[mask4][-i4:]
x_tail4 = x2[mask4][-i4:]

#findings the constants for the power law
slope1, inter1 = stats.linregress(np.log(x_tail1), np.log(M_tail1))[:2]
slope2, inter2 = stats.linregress(np.log(x_tail2), np.log(M_tail2))[:2]
slope3, inter3 = stats.linregress(np.log(x_tail3), np.log(M_tail3))[:2]
slope4, inter4 = stats.linregress(np.log(x_tail4), np.log(M_tail4))[:2]
print('slope1 = ', slope1, 'inter1 =', inter1)
print('slope2 = ', slope2, 'inter2 =', inter2)
print('slope3 = ', slope3, 'inter3 =', inter3)
print('slope4 = ', slope4, 'inter4 =', inter4)

#General analysis plot
plt.figure(figsize=(10,6))
plt.loglog(x2,M1,)
plt.loglog(x2,M2)
plt.loglog(x2,M3)
plt.loglog(x2,M4,)
plt.title('Money per agent. 10^%d transactions. MC = %d' %(int(np.log10(n)), mc))
plt.xlabel('Money')
plt.ylabel('amount of agents')
plt.legend(('α = 0.5', 'α = 1.0', 'α = 1.5', 'α = 2.0'))
plt.xlim(0.1,max)
plt.show()

#Tail analysis 1 plot
plt.figure(figsize=(10,6))
plt.loglog(x2,M1)
plt.title('Tail extraction with α = 0.5.  10^%d transactions. MC = %d' %(int(np.log10(n)), mc))
plt.xlabel('Money')
plt.ylabel('amount of agents')
plt.loglog(x, (np.exp(inter1)*x**(slope1)), color='green')
plt.legend(('Numerical', 'Power law'))
plt.xlim(x_tail1[0],x_tail1[-1])
plt.ylim(M_tail1[0],M_tail1[-1])
plt.show()

#Tail analysis 2 plot
plt.figure(figsize=(10,6))
plt.loglog(x2,M2)
plt.title('Tail extraction with α = 1.0.  10^%d transactions. MC = %d' %(int(np.log10(n)), mc))
plt.xlabel('Money')
plt.ylabel('amount of agents')
plt.loglog(x, (np.exp(inter2)*x**(slope2)), color='green')
plt.legend(('Numerical', 'Power law'))
plt.xlim(x_tail2[0],x_tail2[-1])
plt.ylim(M_tail2[0],M_tail2[-1])
plt.show()

#Tail analysis 3 plot
plt.figure(figsize=(10,6))
plt.loglog(x2,M3)
plt.title('Tail extraction with α = 1.5.  10^%d transactions. MC = %d' %(int(np.log10(n)), mc))
plt.xlabel('Money')
plt.ylabel('amount of agents')
plt.loglog(x, (np.exp(inter3)*x**(slope3)), color='green')
plt.legend(('Numerical', 'Power law'))
plt.xlim(x_tail3[0],x_tail3[-1])
plt.ylim(M_tail3[0],M_tail3[-1])
plt.show()

#Tail analysis 4 plot
plt.figure(figsize=(10,6))
plt.loglog(x2,M4)
plt.title('Tail extraction with α = 2.0.  10^%d transactions. MC = %d' %(int(np.log10(n)), mc))
plt.xlabel('Money')
plt.ylabel('amount of agents')
plt.loglog(x, (np.exp(inter4)*x**(slope4)), color='green')
plt.legend(('Numerical', 'Power law'))
plt.xlim(x_tail4[0],x_tail4[-1])
plt.ylim(M_tail4[0],M_tail4[-1])
plt.show()
