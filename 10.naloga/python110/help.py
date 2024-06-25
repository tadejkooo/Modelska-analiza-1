import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

plt.rcParams.update({'font.size': 12})

import numpy as np

def DFT(s, window='No window', sigma=100.):
    s = np.asarray(s, dtype=float)
    N = s.shape[0]

    if window == 'No window':
        w = np.ones(N, dtype=float)
    elif window == 'Bartlett':
        w = 1. - np.abs(np.arange(N, dtype=float) - N / 2.) / (N / 2.)
    elif window == 'Hann':
        w = 0.5 * (1. - np.cos(2. * np.pi * np.arange(N, dtype=float) / N))
    elif window == 'Welch':
        w = 1. - ((np.arange(N, dtype=float) - N / 2.) / (N / 2.))**2.
    elif window == 'Gauss':
        w = np.exp(-((np.arange(N, dtype=float) - N / 2.) / sigma)**2.)

    # Create a frequency vector
    k = np.arange(N)
    
    # Compute the weight matrix
    wn = np.exp(-1j * 2. * np.pi * np.outer(k, k) / N)

    # Apply the window to the input signal
    sw = w * s

    # Compute the DFT using matrix multiplication
    S = np.dot(wn, sw)

    # Power spectrum
    P = np.abs(S)**2
    
    return S, P


def iDFT(S):
    N = len(S)

    # Create a frequency vector
    k = np.arange(N)
    
    # Compute the weight matrix
    wn = np.exp(1j * 2. * np.pi * np.outer(k, k) / N)

    # Compute the iDFT using matrix multiplication
    s = np.dot(S, wn) / N

    return s

c0 = np.loadtxt('data/signal0.dat')
c1 = np.loadtxt('data/signal1.dat')
c2 = np.loadtxt('data/signal2.dat')
c3 = np.loadtxt('data/signal3.dat')

N = len(c0)

ts = np.arange(0, N, dtype=float)
dt = ts[1] - ts[0]

nuc = 1. / (2. * dt)
nus = np.linspace(-nuc, nuc, N)
dnu = nus[1]-nus[0]

tau = 16.
r = 1. / (2.*tau) * np.exp(-np.abs(ts)/(tau)) # Prenosna funkcija

for i in range(int(len(r)/2)):
    r[-i]=r[i]

C0, P0 = DFT(c0)
C1, P1 = DFT(c1)
C2, P2 = DFT(c2)
C3, P3 = DFT(c3)

R, Pr = DFT(r)

def exponent(t, A, b):
    return A*np.exp(-b*t)

# signal0.dat

meja = 70
N = np.ones_like(P0)*np.average(P0[meja:int(len(P0)/2)])
popt, pcov = curve_fit(exponent, ts[:4], P0[:4], p0=(10000, 1))
A_opt = popt[0]
b_opt = popt[1]
S = exponent(ts, A_opt, b_opt)

fig = plt.figure(figsize=(12, 5))

ax = fig.add_subplot(1, 1, 1)
ax.set_yscale('log')
ax.plot(ts, P0)
ax.plot(ts, N)
ax.plot(ts, S)
ax.set_ylim((10**(-7), 10**(5)))
ax.scatter(ts[10], P0[10], color='red')
plt.show()