# Inverse Problem: Determining {p(t), u(x,t)} in a Parabolic PDE

This repository contains three numerical methods for identifying the functions \( p(t) \) and \( u(x,t) \) in the parabolic PDE:

\[ u_t - u_{xx} + p(t)u = f(x,t), \quad u(0,t) = u(l,t) = 0, \quad \int_0^l u(x,t)\omega(x)dx = g(t) \]

## Methods

1. **Integration Method**: Based on an explicit integral formula.
2. **Newton-Raphson Method**: Iterative root-finding using overdetermination.
3. **Physics-Informed Neural Network (PINN)**: Utilizes neural networks to approximate solutions.

## Repository Structure

- `method1_integration.py`: Implementation of the Integration Method.
- `method2_newton.py`: Implementation of the Newton-Raphson Method.
- `method3_pinn.py`: Implementation of the PINN approach.
- `common/`: Contains shared functions and problem setup.
- `utils/`: Utility functions for tasks like plotting.
- `results/`: Stores output files such as `.npy` results.

## How to Run

Ensure you have the necessary dependencies installed. Then, run any of the methods using:

```bash
python method1_integration.py
python method2_newton.py
python method3_pinn.py
