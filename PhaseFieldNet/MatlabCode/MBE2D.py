import time
import numpy as np
import torch
from GRF import GRF  # Assuming GRF is provided as in the Allen-Cahn example

# Parameter Initialization

# Spatial Parameters
Nx, Ny = 64, 64
Lx, Ly = 2 * np.pi, 2 * np.pi
hx, hy = Lx / Nx, Ly / Ny

x = torch.linspace(-0.5 * Lx + hx, 0.5 * Lx, Nx)
y = torch.linspace(-0.5 * Ly + hy, 0.5 * Ly, Ny)
xx, yy = torch.meshgrid(x, y)

# Constant
epsilon = 0.1

# Discrete Fourier Transform
p = 1j * 2 * np.pi / Lx * torch.cat([torch.arange(0, Nx // 2), torch.tensor([0]), torch.arange(-Nx // 2 + 1, 0)])
q = 1j * 2 * np.pi / Ly * torch.cat([torch.arange(0, Ny // 2), torch.tensor([0]), torch.arange(-Ny // 2 + 1, 0)])
pp, qq = torch.meshgrid(p, q)

p2 = (2 * np.pi / Lx * torch.cat([torch.arange(0, Nx // 2 + 1), torch.arange(-Nx // 2 + 1, 0)]))**2
q2 = (2 * np.pi / Ly * torch.cat([torch.arange(0, Ny // 2 + 1), torch.arange(-Ny // 2 + 1, 0)]))**2
pp2, qq2 = torch.meshgrid(p2, q2)

# Initial Condition (Gaussian Random Field)
tau = 150.0
alpha = 100.0
u = 10 * GRF(alpha, tau, Nx)

# Time Discretization
dt = 0.00025
Nt = 10000
T = Nt * dt
Np = 100
ns = Nt // Np

# Move to GPU if available
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

u = torch.tensor(u, dtype=torch.complex64, device=device)
pp, qq, pp2, qq2 = pp.to(device), qq.to(device), pp2.to(device), qq2.to(device)

# Time Evolution
t_start = time.time()
for iter in range(Nt):
    u = u.real
    tu = torch.fft.fft2(u)

    fx = torch.fft.ifft2(pp * tu).real
    fy = torch.fft.ifft2(qq * tu).real

    f1 = (fx**2 + fy**2) * fx
    f2 = (fx**2 + fy**2) * fy

    s_hat = torch.fft.fft2(u / dt) + pp * torch.fft.fft2(f1) + qq * torch.fft.fft2(f2)
    v_hat = s_hat / (1 / dt - (pp2 + qq2) + epsilon * (pp2 + qq2)**2)
    u = torch.fft.ifft2(v_hat)

elapsed_time = time.time() - t_start
print("Elapsed time = ", elapsed_time)
