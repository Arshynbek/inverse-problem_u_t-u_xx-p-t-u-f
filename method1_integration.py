import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from common.problem_setup import u_exact, p_exact, f_exact, omega, omega_dd, g_exact, gprime_exact
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

# Initial and boundary conditions
u[0, :] = u_exact(x, 0)
u[:, 0] = 0
u[:, -1] = 0

# Initial p(0)
integral_u_omega0 = np.sum(u[0, :] * omega(x)) * h
integral_u_omega_pp0 = np.sum(u[0, :] * omega_dd(x)) * h
integral_f_omega0 = np.sum(f_exact(x, 0) * omega(x)) * h
numerator0 = integral_u_omega_pp0 + integral_f_omega0 - gprime_exact(0)
p_num[0] = numerator0 / g_exact(0)

# Time stepping
for k in range(M):
    u_prev = u[k, :]
    rhs = np.zeros(N - 1)

    for i in range(1, N):
        rhs[i - 1] = (
            (1 - tau / h**2) * u_prev[i]
            + (tau / (2 * h**2)) * (u_prev[i + 1] + u_prev[i - 1])
            + 0.5 * tau * (f_exact(x[i], t[k + 1]) + f_exact(x[i], t[k]))
        )

    p_guess = p_num[k]
    main_diag = (1 + tau / h**2 + tau * p_guess) * np.ones(N - 1)
    off_diag = (-tau / (2 * h**2)) * np.ones(N - 2)
    A = diags([off_diag, main_diag, off_diag], [-1, 0, 1]).tocsc()
    u_next_interior = spsolve(A, rhs)

    u_next = np.zeros(N + 1)
    u_next[1:-1] = u_next_interior
    u[k + 1, :] = u_next

    # Update p
    integral_u_omega = np.sum(u_next * omega(x)) * h
    integral_u_omega_pp = -np.pi**2 * integral_u_omega
    ux0 = (u_next[1] - u_next[0]) / h
    uxl = (u_next[N] - u_next[N - 1]) / h
    boundary_terms = uxl * omega(x[N]) - ux0 * omega(x[0])
    integral_f_omega = np.sum(f_exact(x, t[k + 1]) * omega(x)) * h

    numerator = integral_u_omega_pp + boundary_terms + integral_f_omega - gprime_exact(t[k + 1])
    p_num[k + 1] = numerator / g_exact(t[k + 1])

# Compute and plot errors
p_exact_vals = p_exact(t)
plot_results(t, p_exact_vals, p_num, x, u_exact(x, T), u[-1, :], method_name="Integration")
np.save("results/p_method=1.npy", p_num)
np.save("results/p_exact.npy", p_exact_vals)
