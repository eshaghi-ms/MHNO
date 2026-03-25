import numpy as np
import torch
import time
from GRF import GRF

# Parameter Initialization

# Spatial Parameters
Nx = 64
Ny = Nx
Lx = 64
Ly = Lx
hx = Lx / Nx
hy = Ly / Ny

x = torch.linspace(-0.5 * Lx + hx, 0.5 * Lx, Nx)
y = torch.linspace(-0.5 * Ly + hy, 0.5 * Ly, Ny)

# Constant
epsilon = 0.025

# Discrete Fourier Transform
p = 2 * np.pi / Lx * torch.cat([torch.arange(0, Nx // 2 + 1), torch.arange(-Nx // 2 + 1, 0)])
q = 2 * np.pi / Ly * torch.cat([torch.arange(0, Ny // 2 + 1), torch.arange(-Ny // 2 + 1, 0)])
p2 = p ** 2
q2 = q ** 2
pp2, qq2 = torch.meshgrid(p2, q2)

# Time Discretization
dt = 10
Nt = 100
T = Nt * dt
Np = 100
ns = Nt // Np

# Initial Condition
u_mean = 0.07
tau = 3.5
alpha = 2.0
norm_a = GRF(alpha, tau, Nx)
u = u_mean + u_mean * norm_a

# Convert to PyTorch tensor
u = torch.tensor(u, dtype=torch.float32)

# Move to GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

start_time = time.time()
u = u.to(device)
pp2, qq2 = pp2.to(device), qq2.to(device)

# Update
for iter in range(Nt):
    u = u.real
    s_hat = torch.fft.fft2(u / dt) - (pp2 + qq2) * torch.fft.fft2(u ** 3) + 2 * (pp2 + qq2) ** 2 * torch.fft.fft2(u)
    v_hat = s_hat / (1.0 / dt + (1 - epsilon) * (pp2 + qq2) + (pp2 + qq2) ** 3)
    u = torch.fft.ifft2(v_hat)
    u = torch.real(u)

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.4f} seconds")
