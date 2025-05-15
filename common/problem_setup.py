import numpy as np

def omega(x):
    return np.sin(np.pi * x)

def omega_dd(x):
    return -np.pi**2 * np.sin(np.pi * x)

def u_exact(x, t):
    return np.exp(t) * np.sin(np.pi * x)

def p_exact(t):
    return np.exp(-t)

def f_exact(x, t):
    return np.sin(np.pi * x) * (1 + (1 + np.pi**2) * np.exp(t))

def g_exact(t):
    return 0.5 * np.exp(t)

def gprime_exact(t):
    return 0.5 * np.exp(t)
