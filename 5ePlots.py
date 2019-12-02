#5e) Tail Plots:
import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import special, stats
plt.rcParams.update({'font.size': 14})

max = 100 #max value for money / x-value
n = 10**6 #1024 for the degree plot, 10**7 otherwise
N = 1000 #1000
m0 = 1 #starter money
mc = 1000 #10**3 - 10**4
dm = 0.01 #bins width
beta = 1/m0
lam = 0 #0, 0.25

#lam = 0: 'outfile2.npy'
#loading calculations. with lam=0.25 is outfile2_lam.npy
(M10,M11,M12,M13,M14,M20,M21,M22,M23,M24) = np.load('outfile2.npy')

x = np.linspace(0,max,N)
x2 = np.linspace(0,max,len(M10))
#intervals to look at for tails. lam=0
i10 = 500; i11 = 300; i12 = 300; i13 = 300; i14 = 300
i20 = 2000; i21 = 800; i22 = 400; i23 = 300; i24 = 300
'''
#intervals to look at for tails. lam=0.25
i10 = 500; i11 = 300; i12 = 300; i13 = 300; i14 = 300
i20 = 2000; i21 = 800; i22 = 400; i23 = 300; i24 = 300
'''

#masks for different alphas, gammas
mask10 = np.where(M10 != 0) # only looking at the values where agents have money
M_tail10 = M10[mask10][-i10:]
x_tail10 = x2[mask10][-i10:]
mask11 = np.where(M11 != 0)
M_tail11 = M11[mask11][-i11:]
x_tail11 = x2[mask11][-i11:]
mask12 = np.where(M12 != 0)
M_tail12 = M12[mask12][-i12:]
x_tail12 = x2[mask12][-i12:]
mask13 = np.where(M13 != 0)
M_tail13 = M13[mask13][-i13:]
x_tail13 = x2[mask13][-i13:]
mask14 = np.where(M14 != 0)
M_tail14 = M14[mask14][-i14:]
x_tail14 = x2[mask14][-i14:]

mask20 = np.where(M20 != 0) # only looking at the values where agents have money
M_tail20 = M20[mask20][-i20:]
x_tail20 = x2[mask20][-i20:]
mask21 = np.where(M21 != 0)
M_tail21 = M21[mask21][-i21:]
x_tail21 = x2[mask21][-i21:]
mask22 = np.where(M22 != 0)
M_tail22 = M22[mask22][-i22:]
x_tail22 = x2[mask22][-i22:]
mask23 = np.where(M23 != 0)
M_tail23 = M23[mask23][-i23:]
x_tail23 = x2[mask23][-i23:]
mask24 = np.where(M24 != 0)
M_tail24 = M24[mask24][-i24:]
x_tail24 = x2[mask24][-i24:]

#findings the constants for the power law
slope10, inter10 = stats.linregress(np.log(x_tail10), np.log(M_tail10))[:2]
slope11, inter11 = stats.linregress(np.log(x_tail11), np.log(M_tail11))[:2]
slope12, inter12 = stats.linregress(np.log(x_tail12), np.log(M_tail12))[:2]
slope13, inter13 = stats.linregress(np.log(x_tail13), np.log(M_tail13))[:2]
slope14, inter14 = stats.linregress(np.log(x_tail14), np.log(M_tail14))[:2]
print('slope10 = ', slope10, 'inter10 =', inter10)
print('slope11 = ', slope11, 'inter11 =', inter11)
print('slope12 = ', slope12, 'inter12 =', inter12)
print('slope13 = ', slope13, 'inter13 =', inter13)
print('slope14 = ', slope14, 'inter14 =', inter14)
slope20, inter20 = stats.linregress(np.log(x_tail20), np.log(M_tail20))[:2]
slope21, inter21 = stats.linregress(np.log(x_tail21), np.log(M_tail21))[:2]
slope22, inter22 = stats.linregress(np.log(x_tail22), np.log(M_tail22))[:2]
slope23, inter23 = stats.linregress(np.log(x_tail23), np.log(M_tail23))[:2]
slope24, inter24 = stats.linregress(np.log(x_tail24), np.log(M_tail24))[:2]
print('slope20 = ', slope20, 'inter20 =', inter20)
print('slope21 = ', slope21, 'inter21 =', inter21)
print('slope22 = ', slope22, 'inter22 =', inter22)
print('slope23 = ', slope23, 'inter23 =', inter23)
print('slope24 = ', slope24, 'inter24 =', inter24)

#Tail analysis 10 plot
plt.figure(figsize=(10,6))
plt.loglog(x2,M10)
plt.title('Tail extraction with α = 1.0, γ = 0.0.  10^%d transactions. MC = %d' %(int(np.log10(n)), mc))
plt.xlabel('Money')
plt.ylabel('amount of agents')
plt.loglog(x, (np.exp(inter10)*x**(slope10)), color='green')
plt.legend(('Numerical', 'Power law'))
plt.xlim(x_tail10[0],x_tail10[-1])
plt.ylim(M_tail10[0],M_tail10[-1])
plt.show()

#Tail analysis 11 plot
plt.figure(figsize=(10,6))
plt.loglog(x2,M11)
plt.title('Tail extraction with α = 1.0, γ = 1.0.  10^%d transactions. MC = %d' %(int(np.log10(n)), mc))
plt.xlabel('Money')
plt.ylabel('amount of agents')
plt.loglog(x, (np.exp(inter11)*x**(slope11)), color='green')
plt.legend(('Numerical', 'Power law'))
plt.xlim(x_tail11[0],x_tail11[-1])
plt.ylim(M_tail11[0],M_tail11[-1])
plt.show()

#Tail analysis 12 plot
plt.figure(figsize=(10,6))
plt.loglog(x2,M12)
plt.title('Tail extraction with α = 1.0, γ = 2.0.  10^%d transactions. MC = %d' %(int(np.log10(n)), mc))
plt.xlabel('Money')
plt.ylabel('amount of agents')
plt.loglog(x, (np.exp(inter12)*x**(slope12)), color='green')
plt.legend(('Numerical', 'Power law'))
plt.xlim(x_tail12[0],x_tail12[-1])
plt.ylim(M_tail12[0],M_tail12[-1])
plt.show()

#Tail analysis 13 plot
plt.figure(figsize=(10,6))
plt.loglog(x2,M13)
plt.title('Tail extraction with α = 1.0, γ = 3.0.  10^%d transactions. MC = %d' %(int(np.log10(n)), mc))
plt.xlabel('Money')
plt.ylabel('amount of agents')
plt.loglog(x, (np.exp(inter13)*x**(slope13)), color='green')
plt.legend(('Numerical', 'Power law'))
plt.xlim(x_tail13[0],x_tail13[-1])
plt.ylim(M_tail13[0],M_tail13[-1])
plt.show()

#Tail analysis 14 plot
plt.figure(figsize=(10,6))
plt.loglog(x2,M14)
plt.title('Tail extraction with α = 1.0, γ = 4.0.  10^%d transactions. MC = %d' %(int(np.log10(n)), mc))
plt.xlabel('Money')
plt.ylabel('amount of agents')
plt.loglog(x, (np.exp(inter14)*x**(slope14)), color='green')
plt.legend(('Numerical', 'Power law'))
plt.xlim(x_tail14[0],x_tail14[-1])
plt.ylim(M_tail14[0],M_tail14[-1])
plt.show()

#Tail analysis 20 plot
plt.figure(figsize=(10,6))
plt.loglog(x2,M20)
plt.title('Tail extraction with α = 2.0, γ = 0.0.  10^%d transactions. MC = %d' %(int(np.log10(n)), mc))
plt.xlabel('Money')
plt.ylabel('amount of agents')
plt.loglog(x, (np.exp(inter20)*x**(slope20)), color='green')
plt.legend(('Numerical', 'Power law'))
plt.xlim(x_tail20[0],x_tail20[-1])
plt.ylim(M_tail20[0],M_tail20[-1])
plt.show()

#Tail analysis 21 plot
plt.figure(figsize=(10,6))
plt.loglog(x2,M21)
plt.title('Tail extraction with α = 2.0, γ = 1.0.  10^%d transactions. MC = %d' %(int(np.log10(n)), mc))
plt.xlabel('Money')
plt.ylabel('amount of agents')
plt.loglog(x, (np.exp(inter21)*x**(slope21)), color='green')
plt.legend(('Numerical', 'Power law'))
plt.xlim(x_tail21[0],x_tail21[-1])
plt.ylim(M_tail21[0],M_tail21[-1])
plt.show()

#Tail analysis 22 plot
plt.figure(figsize=(10,6))
plt.loglog(x2,M22)
plt.title('Tail extraction with α = 2.0, γ = 2.0.  10^%d transactions. MC = %d' %(int(np.log10(n)), mc))
plt.xlabel('Money')
plt.ylabel('amount of agents')
plt.loglog(x, (np.exp(inter22)*x**(slope22)), color='green')
plt.legend(('Numerical', 'Power law'))
plt.xlim(x_tail22[0],x_tail22[-1])
plt.ylim(M_tail22[0],M_tail22[-1])
plt.show()

#Tail analysis 23 plot
plt.figure(figsize=(10,6))
plt.loglog(x2,M23)
plt.title('Tail extraction with α = 2.0, γ = 3.0.  10^%d transactions. MC = %d' %(int(np.log10(n)), mc))
plt.xlabel('Money')
plt.ylabel('amount of agents')
plt.loglog(x, (np.exp(inter23)*x**(slope23)), color='green')
plt.legend(('Numerical', 'Power law'))
plt.xlim(x_tail23[0],x_tail23[-1])
plt.ylim(M_tail23[0],M_tail23[-1])
plt.show()

#Tail analysis 24 plot
plt.figure(figsize=(10,6))
plt.loglog(x2,M24)
plt.title('Tail extraction with α = 2.0, γ = 4.0.  10^%d transactions. MC = %d' %(int(np.log10(n)), mc))
plt.xlabel('Money')
plt.ylabel('amount of agents')
plt.loglog(x, (np.exp(inter24)*x**(slope24)), color='green')
plt.legend(('Numerical', 'Power law'))
plt.xlim(x_tail24[0],x_tail24[-1])
plt.ylim(M_tail24[0],M_tail24[-1])
plt.show()
