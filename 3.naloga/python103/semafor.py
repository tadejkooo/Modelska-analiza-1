import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize, Bounds

plt.rcParams.update({'font.size': 12})

N = 20 # st tock
taus = np.linspace(0,1,N)

nu0 = 2.0
kappa1 = 50.

bounds = [(nu0-0.1, nu0+0.1)] + [(-10., 10.) for _ in range(N-1)]
bounds = [(-np.inf, np.inf) for _ in range(N)]
bounds = Bounds(*zip(*bounds))
deltau = taus[1]-taus[0]

def func1(nus):
    F = 0.
    a1 = 0.

    F += 1./2. * (nus[0] / deltau)**2.
    a1 += 1./2. * nus[0]

    for i in range(1, len(nus)-1):
        
        F += ((nus[i]-nus[i-1]) / deltau)**2.
        a1 += nus[i]

    F += 1./2. * ((nus[-1]-nus[-2]) / deltau)**2.
    a1 += 1./2. * nus[-1]
    a1 += - 1.
    v1 = 1 + np.exp(kappa1*a1)

    return F + v1

initial_guess = np.random.rand(N)

result = minimize(func1, x0=initial_guess, method='COBYLA', bounds=bounds)

print(result)
opt_nus = result.x
print(opt_nus)

plt.plot(taus, opt_nus)

plt.show()