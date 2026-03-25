# vectorized version

import time
import numpy as np
import torch
from GRF import GRF  # Assuming GRF function is provided
from plotting import plot_U

# Parameter Initialization
np.random.seed(42)
num_samples = 100
plot_indices = [0, 10, 20, 50]

# Spatial Parameters
Nx, Ny = 64, 64
Lx, Ly = 1, 1
hx, hy = Lx / Nx, Ly / Ny

x = torch.linspace(-0.5 * Lx + hx, 0.5 * Lx, Nx)
y = torch.linspace(-0.5 * Ly + hy, 0.5 * Ly, Ny)
xx, yy = torch.meshgrid(x, y, indexing='ij')

# Interfacial energy constant
epsilon = 0.0125
Cahn = epsilon ** 2

# Discrete Fourier Transform
p = 2 * np.pi / Lx * torch.cat([torch.arange(0, Nx // 2 + 1), torch.arange(-Nx // 2 + 1, 0)])
q = 2 * np.pi / Ly * torch.cat([torch.arange(0, Ny // 2 + 1), torch.arange(-Ny // 2 + 1, 0)])
p2 = p ** 2
q2 = q ** 2
pp2, qq2 = torch.meshgrid(p2, q2, indexing='ij')

# Time Discretization
dt = 0.0025
T = 0.5
Nt = round(T / dt)
Np = 200
ns = Nt // Np

# Initial Condition
tau = 2000
alpha = 1

U = np.zeros((num_samples, Nx, Nx))
for i in range(num_samples):
    U[i,:,:] = GRF(alpha, tau, Nx)  # Initial condition using the GRF function
U = torch.tensor(U, dtype=torch.float32)

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
U = U.to(device)
pp2, qq2 = pp2.to(device), qq2.to(device)

plot_U(U, plot_indices, 'Start')
# Time Evolution
t_start = time.time()
for iteration in range(Nt):
    s_hat = torch.fft.fft2(U) - dt * (pp2 + qq2) * torch.fft.fft2(U ** 3 - 3 * U)
    v_hat = s_hat / (1.0 + dt * (2.0 * (pp2 + qq2) + Cahn * (pp2 + qq2) ** 2))
    U = torch.fft.ifft2(v_hat)
    U = torch.real(U)

Dt = time.time() - t_start
print("Elapsed time for", num_samples, "samples and", Nt, "time steps =", Dt)
plot_U(U, plot_indices, 'End')
