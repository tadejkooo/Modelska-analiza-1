import numpy as np
import pandas as pd

from scipy.linalg import toeplitz


def DFT(s, window='No window', sigma=100.): # Fourierova transformacija

    s = np.asarray(s, dtype=float)
    N = s.shape[0]

    if window == 'No window':
        w = np.ones(N, dtype=float)

    elif window == 'Bartlett':
        w = 1. - np.abs(np.arange(N, dtype=float) - float(N)/2.) / (float(N)/2.)

    elif window == 'Hann':
        w = 1./2. * (1. - np.cos(2.*np.pi*np.arange(N, dtype=float)/float(N)))

    elif window == 'Welch':
        w = 1. - ((np.arange(N, dtype=float) - float(N)/2.) / (float(N)/2.))**2.

    elif window == 'Gauss':
        w = np.exp(-((np.arange(N, dtype=float) - float(N)/2.)/sigma)**2.)

    S = np.zeros_like(s, dtype=complex)

    for i in range(0, N):
        a = 0.

        for k in range(N):
            a += w[k]*s[k]*np.exp(-1j * 2 * np.pi * k * i / N)

        S[i] = a

    # Power spectrum
    W = np.sum(w**2)
    P = np.zeros(round(N/2))
    P[0] = 1./W * np.abs(S[0])**2
    # P[-1] = 1./W * np.abs(S[round(N/2)])**2
    
    for i in range(1, len(P)):
        P[i] = 1./(2.*W) * (np.abs(S[i])**2 + np.abs(S[N-i])**2)

    return S, P


def autocorrelation(s):

    N = len(s)
    R = []

    for k in range(N):
        numerator = sum(s[n] * s[n - k] for n in range(k, N))
        R_k = numerator / (N - k)
        R.append(R_k)

    return np.array(R)


def solve_toeplitz_system(R, p):

    # Create the Toeplitz matrix M
    c = R[:p]
    r = R[:p]
    M = toeplitz(c, r)
    # Create the right-hand side vector b
    b = R[1:p + 1]
    # Solve the linear system
    a = np.linalg.solve(M, -b)

    return a


def find_poles(s, p):
    """
    Calculates the poles of the passing function
    """
    
    R = autocorrelation(s)
    a = solve_toeplitz_system(R, p)
    # Construct the polynomial coefficients
    coeffs = [1] + list(a)  # Coefficients of z^0, z^1, ..., z^p
    # Find the zeros of the polynomial (poles of the passing function)
    poles = np.roots(coeffs)

    return poles


def adjust_poles_and_recompute_a(s, p, type):
    """
    Adjusts the poles by reflecting those outside the unit circle and recomputes the coefficients.
    
    Parameters:
    s (array-like): Signal data.
    p (int): Model order (number of reflection coefficients).
    
    Returns:
    ndarray: Adjusted coefficients a_k.
    """
    # Step 1: Compute initial poles
    R = autocorrelation(s)
    a = solve_toeplitz_system(R, p)
    
    # Construct the polynomial coefficients
    coeffs = [1] + list(a)  # Coefficients of z^0, z^1, ..., z^p
    poles = np.roots(coeffs)

    if type=='Reflect':
        # Step 2: Reflect poles outside the unit circle
        poles_adjusted = np.array([z if np.abs(z) <= 1 else 0.99 / np.conj(z) for z in poles])

    if type=='Project':
        # Step 2: Project poles outside the unit circle
        poles_adjusted = np.array([z if np.abs(z) <= 1 else z / np.abs(z) for z in poles])

    # Step 3: Recompute coefficients from adjusted poles
    coeffs_adjusted = np.poly(poles_adjusted)  # Get polynomial coefficients
    a_adjusted = coeffs_adjusted[1:]  # Exclude the leading 1 for z^0
    
    return poles, poles_adjusted, a_adjusted


def MESE(omega, s, p, type):
    """
    Maximum Entropy Spectral Estimation (MESE) with adjusted poles.
    
    Parameters:
    omega (array-like): Frequencies (in radians) to compute the spectrum.
    s (array-like): Signal data.
    p (int): Model order (number of reflection coefficients).
    type (str): type of adjustment of poles
    
    Returns:
    ndarray: Spectral power at the given frequencies.
    """
    # Adjust poles and recompute coefficients
    poles, poles_adjusted, a_adjusted = adjust_poles_and_recompute_a(s, p, type)
    
    # Compute signal power G^2
    R = autocorrelation(s)
    G2 = np.abs(R[0] + np.sum(a_adjusted * R[1:p+1]))
    
    # Compute the denominator for all frequencies
    exp_term = np.exp(-1j * omega[:, None] * np.arange(1, p+1))  # Shape: (len(omega), p)
    denominator = np.abs(1 + np.sum(a_adjusted * exp_term, axis=1))**2
    
    # Compute the spectral power
    P = G2 / denominator
    return poles, poles_adjusted, P


def predict(s, p, num, type):
    """
    Linear prediction of the future signal
    """

    poles, poles_adjusted, a = adjust_poles_and_recompute_a(s, p, type)

    for _ in range(num):
        sn = - np.sum(a*s[-p:][::-1])
        s = np.append(s, sn)
    
    return poles, poles_adjusted, s