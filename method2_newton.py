import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from common.problem_setup import u_exact, p_exact, f_exact, omega, g_exact
from utils.plotting import plot_results

# Parameters
l = 1.0
T = 1.0
N = 100
M = 200
h = l / N
tau = T / M
x = np.linspace(0, l, N + 1)
t = np.linspace(0, T, M + 1)

u = np.zeros((M + 1, N + 1))
p_num = np.zeros(M + 1)
u[0, :] = u_exact(x, 0)
u[:, 0] = 0
u[:, -1] = 0

omega_weights = omega(x)[1:-1] * h
p_num[0] = 1.0

# Time stepping with Newton-Raphson
for k in range(M):
    u_prev = u[k, :]
    rhs = np.zeros(N - 1)
    for i in range(1, N):
        rhs[i - 1] = (
            (1 - tau / h**2) * u_prev[i]
            + (tau / (2 * h**2)) * (u_prev[i + 1] + u_prev[i - 1])
            + 0.5 * tau * (f_exact(x[i], t[k]) + f_exact(x[i], t[k + 1]))
        )

    p_guess = p_num[k]
    tol = 1e-8
    for iteration in range(20):
        main_diag = (1 + tau / h**2 + tau * p_guess) * np.ones(N - 1)
        off_diag = (-tau / (2 * h**2)) * np.ones(N - 2)
        A = diags([off_diag, main_diag, off_diag], [-1, 0, 1]).tocsc()
        u_interior = spsolve(A, rhs)

        u_next = np.zeros(N + 1)
        u_next[1:-1] = u_interior
        residual = np.dot(u_next[1:-1], omega_weights) - g_exact(t[k + 1])
        if abs(residual) < tol:
            break

        d_rhs = -tau * u_interior
        dFdp = np.dot(spsolve(A, d_rhs), omega_weights)
        p_guess -= residual / dFdp

    p_num[k + 1] = p_guess
    u[k + 1, :] = u_next

# Errors and plot
p_exact_vals = p_exact(t)
plot_results(t, p_exact_vals, p_num, x, u_exact(x, T), u[-1, :], method_name="Newton")
np.save("results/p_method=2.npy", p_num)
