# vectorized version
import time
import numpy as np
import torch
from GRF import GRF  # Assuming GRF is provided as in the Allen-Cahn example
from plotting import plot_U

# Parameter Initialization
np.random.seed(42)
num_samples = 100
plot_indices = [0, 10, 20, 50]

# Spatial Parameters
Nx, Ny = 64, 64
Lx, Ly = 2 * np.pi, 2 * np.pi
hx, hy = Lx / Nx, Ly / Ny

x = torch.linspace(-0.5 * Lx + hx, 0.5 * Lx, Nx)
y = torch.linspace(-0.5 * Ly + hy, 0.5 * Ly, Ny)
xx, yy = torch.meshgrid(x, y, indexing='ij')

# Constant
epsilon = 0.1

# Discrete Fourier Transform
p = 1j * 2 * np.pi / Lx * torch.cat([torch.arange(0, Nx // 2), torch.tensor([0]), torch.arange(-Nx // 2 + 1, 0)])
q = 1j * 2 * np.pi / Ly * torch.cat([torch.arange(0, Ny // 2), torch.tensor([0]), torch.arange(-Ny // 2 + 1, 0)])
pp, qq = torch.meshgrid(p, q, indexing='ij')

p2 = (2 * np.pi / Lx * torch.cat([torch.arange(0, Nx // 2 + 1), torch.arange(-Nx // 2 + 1, 0)]))**2
q2 = (2 * np.pi / Ly * torch.cat([torch.arange(0, Ny // 2 + 1), torch.arange(-Ny // 2 + 1, 0)]))**2
pp2, qq2 = torch.meshgrid(p2, q2, indexing='ij')

# Initial Condition (Gaussian Random Field)
tau = 150.0
alpha = 100.0
U = np.zeros((num_samples, Nx, Nx))
for i in range(num_samples):
    u = 10 * GRF(alpha, tau, Nx)
    U[i,:,:] = u
    
# Time Discretization
dt = 0.00005
Nt = 40000
T = Nt * dt
Np = 100
ns = Nt // Np

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

U = torch.tensor(U, dtype=torch.complex64, device=device)
pp, qq, pp2, qq2 = pp.to(device), qq.to(device), pp2.to(device), qq2.to(device)
plot_U(U.real, plot_indices, 'Start', cmap='jet')

# Time Evolution
t_start = time.time()
for iter in range(Nt):
    tu = torch.fft.fft2(U)

    fx = torch.fft.ifft2(pp * tu).real
    fy = torch.fft.ifft2(qq * tu).real

    f1 = (fx**2 + fy**2) * fx
    f2 = (fx**2 + fy**2) * fy

    s_hat = torch.fft.fft2(U / dt) + pp * torch.fft.fft2(f1) + qq * torch.fft.fft2(f2)
    v_hat = s_hat / (1 / dt - (pp2 + qq2) + epsilon * (pp2 + qq2)**2)
    U = torch.fft.ifft2(v_hat)
    U = U.real
elapsed_time = time.time() - t_start

print("Elapsed time for", num_samples, "samples and", Nt, "time steps =", elapsed_time)
plot_U(U, plot_indices, 'End', cmap='jet')
