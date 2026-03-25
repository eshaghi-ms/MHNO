import time
import torch
import numpy as np
from GRF import GRF  # Assuming you have this function as in the Allen-Cahn case.

# Parameter Initialization

# Spatial Parameters
Nx = 64
Ny = Nx
Lx = 64.0
Ly = Lx
hx = Lx / Nx
hy = Ly / Ny

x = torch.linspace(-0.5 * Lx + hx, 0.5 * Lx, Nx)
y = torch.linspace(-0.5 * Ly + hy, 0.5 * Ly, Ny)

# Constants
epsilon = 0.5

# Discrete Fourier Transform
p = 2 * np.pi / Lx * torch.cat([torch.arange(0, Nx // 2 + 1), torch.arange(-Nx // 2 + 1, 0)])
q = 2 * np.pi / Ly * torch.cat([torch.arange(0, Ny // 2 + 1), torch.arange(-Ny // 2 + 1, 0)])
p2 = p ** 2
q2 = q ** 2
pp2, qq2 = torch.meshgrid(p2, q2)

# Time Discretization
dt = 0.5
Nt = 100
T = Nt * dt
Np = 100
ns = Nt // Np

# Initial Condition
tau = 8.0
alpha = 4.0
norm_a = GRF(alpha, tau, Nx)
norm_a = norm_a - 0.5 * np.std(norm_a)

u = np.zeros((Nx, Ny))
u[norm_a >= 0] = 1
u[norm_a < 0] = -1
u = torch.tensor(u)

# Move to GPU if available
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

t_start = time.time()
u = u.to(device)
pp2, qq2 = pp2.to(device), qq2.to(device)

# Update Loop
for iteration in range(Nt):
    u = torch.real(u)
    s_hat = torch.fft.fft2(u / dt) - torch.fft.fft2(u ** 3) + 2 * (pp2 + qq2) * torch.fft.fft2(u)
    v_hat = s_hat / (1.0 / dt + (1 - epsilon) + (pp2 + qq2) ** 2)
    u = torch.fft.ifft2(v_hat)
    u = torch.real(u)

elapsed_time = time.time() - t_start
print(f"Elapsed time: {elapsed_time:.4f} seconds")
