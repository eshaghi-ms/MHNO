# vectorized version

import numpy as np
import torch
import time
from GRF import GRF
from plotting import plot_U

# Parameter Initialization
np.random.seed(42)
num_samples = 100
plot_indices = [0, 10, 20, 50]
plot_times = [0, 5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 70, 80, 90, 99]

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
dt = 0.1
Nt = 10000
T = Nt * dt
Np = 100
ns = Nt // Np

# Initial Condition
u_mean = 0.07
tau = 3.5
alpha = 2.0
U = np.zeros((num_samples, Nx, Nx))
for i in range(num_samples):
    norm_a = GRF(alpha, tau, Nx)
    u = u_mean + u_mean * norm_a
    U[i, :, :] = u
# Convert to PyTorch tensor
U = torch.tensor(U, dtype=torch.float32)

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

start_time = time.time()
U = U.to(device)
pp2, qq2 = pp2.to(device), qq2.to(device)
plot_U(U, plot_indices, 'Start', cmap='jet')

# Update
for iter in range(Nt):    
    s_hat = torch.fft.fft2(U / dt) - (pp2 + qq2) * torch.fft.fft2(U ** 3) + 2 * (pp2 + qq2) ** 2 * torch.fft.fft2(U)
    v_hat = s_hat / (1.0 / dt + (1 - epsilon) * (pp2 + qq2) + (pp2 + qq2) ** 3)
    U = torch.fft.ifft2(v_hat)
    U = torch.real(U)

elapsed_time = time.time() - start_time
print("Elapsed time for", num_samples, "samples and", Nt, "time steps =", elapsed_time)
plot_U(U, plot_indices, 'End', cmap='jet')