import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from GRF import GRF

# Parameter Initialization

# Spatial Parameters
Nx = 64
Ny = 64
Lx = 2 * np.pi
Ly = 2 * np.pi
hx = Lx / Nx
hy = Ly / Ny

x = torch.linspace(-0.5 * Lx + hx, 0.5 * Lx, Nx)
y = torch.linspace(-0.5 * Ly + hy, 0.5 * Ly, Ny)
xx, yy = torch.meshgrid(x, y)

# Interfacial energy constant
epsilon = 0.1
Cahn = epsilon ** 2

# Discrete Fourier Transform
p = 2 * np.pi / Lx * torch.cat([torch.arange(0, Nx // 2 + 1), torch.arange(-Nx // 2 + 1, 0)])
q = 2 * np.pi / Ly * torch.cat([torch.arange(0, Ny // 2 + 1), torch.arange(-Ny // 2 + 1, 0)])
p2 = p ** 2
q2 = q ** 2
pp2, qq2 = torch.meshgrid(p2, q2)

# Time Discretization
dt = 0.01
T = 1
Nt = round(T / dt)
ns = 1

# Initial Condition
tau = 400
alpha = 115
norm_a = GRF(alpha, tau, Nx)
norm_a = norm_a - 0.5 * np.std(norm_a)

# Create the u matrix with -1 and 1 values based on GRF outcome
u = np.zeros((Nx, Ny))
u[norm_a >= 0] = 1
u[norm_a < 0] = -1
u = torch.tensor(u)

# Move to GPU if available
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')

U = torch.zeros(Nt, Nx, Ny, device=device)

t_start = time.time()
u = u.to(device)
pp2, qq2 = pp2.to(device), qq2.to(device)

# Update
for iteration in range(Nt):
    s_hat = torch.fft.fft2(Cahn * u - dt * (u ** 3 - 3 * u))
    v_hat = s_hat / (Cahn + dt * (2 + Cahn * (pp2 + qq2)))
    u = torch.fft.ifft2(v_hat)
    u = torch.real(u)
    U[iteration] = u

Dt = time.time() - t_start
print("elapsed time = ", Dt)
