import numpy as np
import torch
from torch import nn
from torch.autograd import grad
import matplotlib.pyplot as plt
from common.problem_setup import u_exact, p_exact

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
            nn.Linear(64, 1)
        )
    def forward(self, x, t):
        xt = torch.cat([x, t], dim=1)
        return x * (1 - x) * t * self.net(xt) + (1 - t) * torch.sin(np.pi * x)

class PNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )
    def forward(self, t): return self.net(t)

u_net = UNet().to(device)
p_net = PNet().to(device)
optimizer_adam = torch.optim.Adam(list(u_net.parameters()) + list(p_net.parameters()), lr=1e-3)
optimizer_lbfgs = torch.optim.LBFGS(list(u_net.parameters()) + list(p_net.parameters()), max_iter=500)

# Training data
N_f = 1000
x_f = torch.rand(N_f, 1, requires_grad=True).to(device)
t_f = torch.rand(N_f, 1, requires_grad=True).to(device)
x_int = torch.linspace(0, 1, 200).view(-1, 1).to(device)
omega = torch.sin(np.pi * x_int)
int_times = [0.25, 0.5, 0.75, 1.0]
λ = 46.0

def closure():
    optimizer_lbfgs.zero_grad()
    u = u_net(x_f, t_f)
    u_t = grad(u, t_f, torch.ones_like(u), create_graph=True)[0]
    u_x = grad(u, x_f, torch.ones_like(u), create_graph=True)[0]
    u_xx = grad(u_x, x_f, torch.ones_like(u_x), create_graph=True)[0]
    p = p_net(t_f)
    f = torch.sin(np.pi * x_f) * (1 + (1 + np.pi**2) * torch.exp(t_f))
    res = u_t - u_xx + p * u - f
    loss_pde = torch.mean(res**2)

    loss_int = 0
    for tj in int_times:
        t_val = torch.full_like(x_int, tj, requires_grad=True)
        u_val = u_net(x_int, t_val)
        integral = torch.trapz(u_val * omega, x_int, dim=0)
        loss_int += (integral - 0.5 * torch.exp(torch.tensor(tj)))**2
    loss_int /= len(int_times)

    loss = loss_pde + λ * loss_int
    loss.backward()
    return loss

# Training
for epoch in range(5000):
    optimizer_adam.zero_grad()
    loss = closure()
    optimizer_adam.step()
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4e}")

print("Starting L-BFGS optimization...")
optimizer_lbfgs.step(closure)

# Evaluate
x_eval = torch.linspace(0, 1, 100).view(-1, 1).to(device)
u_pred = u_net(x_eval, torch.ones_like(x_eval)).detach().cpu().numpy().flatten()
u_true = u_exact(x_eval.cpu().numpy().flatten(), 1.0)
t_eval = torch.linspace(0, 1, 201).view(-1, 1).to(device)
p_pred = p_net(t_eval).detach().cpu().numpy().flatten()
p_true = p_exact(t_eval.cpu().numpy())

print("Max error in u:", np.max(np.abs(u_pred - u_true)))
print("Max error in p:", np.max(np.abs(p_pred - p_true)))
np.save("results/p_method=3.npy", p_pred)

# Plot
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(t_eval.cpu(), p_true, label="Exact")
plt.plot(t_eval.cpu(), p_pred, 'r--', label="PINN")
plt.title("p(t)")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(x_eval.cpu(), u_true, label="Exact")
plt.plot(x_eval.cpu(), u_pred, 'ro', label="PINN")
plt.title("u(x,1)")
plt.legend()
plt.tight_layout()
plt.show()
