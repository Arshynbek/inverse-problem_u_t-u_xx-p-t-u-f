import matplotlib.pyplot as plt
import numpy as np

def plot_results(t, p_exact_vals, p_num, x, u_exact_T, u_numeric_T, method_name):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(t, p_exact_vals, 'b-', label="Exact")
    plt.plot(t, p_num, 'r--', linewidth=2.5, label="Numerical")
    plt.xlabel('t')
    plt.ylabel('p(t)')
    plt.title(f'Coefficient p(t) - {method_name}')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(x, u_exact_T, 'b-', label='Exact')
    plt.plot(x, u_numeric_T, 'ro', label='Numerical')
    plt.xlabel('x')
    plt.ylabel('u(x,T)')
    plt.title(f'Solution u(x,T) - {method_name}')
    plt.legend()

    plt.tight_layout()
    plt.show()
