# inverse-problem_u_t-u_xx-p-t-u-f
Three numerical methods for identifying {p(t), u(x,t)} in a parabolic PDE with overdetermination
This repository contains three numerical approaches to identify the unknown function \( p(t) \) and the solution \( u(x,t) \) in the parabolic PDE:

\[
u_t - u_{xx} + p(t)u = f(x,t), \quad u(0,t) = u(l,t) = 0, \quad \int_0^l u(x,t)\omega(x)\,dx = g(t)
\]

## Methods

1. **Integration Method** – based on an explicit integral formula
2. **Newton-Raphson Method** – iterative root-finding using overdetermination
3. **PINN** – Physics-Informed Neural Network

## Structure

- `method1_integration.py`
- `method2_newton.py`
- `method3_pinn.py`
- `common/problem_setup.py` – contains exact functions
- `results/` – stores `.npy` outputs

## How to Run

```bash
python method1_integration.py
python method2_newton.py
python method3_pinn.py
