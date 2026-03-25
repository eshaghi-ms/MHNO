import numpy as np
import torch
from scipy.fftpack import dctn, idctn
import time

# Parameter Initialization

# Spatial Parameters
nx, ny = 64, 64
xL, xR = -3, 3
yL, yR = -3, 3
h = (xR - xL) / nx

x = np.linspace(xL + 0.5 * h, xR - 0.5 * h, nx)
y = np.linspace(yL + 0.5 * h, yR - 0.5 * h, ny)

# Time Discretization
T = 0.5
dt = 0.0005
nt = int(T / dt)

# Discrete Fourier Transform (DCT) Frequencies
xi = np.pi * np.arange(0, nx) / (xR - xL)
eta = np.pi * np.arange(0, ny) / (yR - yL)
xi2, eta2 = np.meshgrid(xi ** 2, eta ** 2, indexing="ij")

# Constants
eps1 = 0.07
eps2 = eps1 ** 2
sig = 1.0  # Assuming some value for `sig` as it is undefined in the MATLAB code

# Initial Condition (set `psi` and `bar_psi` accordingly)
psi = np.random.rand(nx, ny) * 2 - 1  # Random values in [-1, 1]
bar_psi = np.mean(psi) * np.ones((nx, ny))

# Move arrays to PyTorch for GPU computation if necessary
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
psi = torch.tensor(psi, device=device, dtype=torch.float32)
bar_psi = torch.tensor(bar_psi, device=device, dtype=torch.float32)
xi2 = torch.tensor(xi2, device=device, dtype=torch.float32)
eta2 = torch.tensor(eta2, device=device, dtype=torch.float32)

# Time-Stepping Loop
t_start = time.time()
for it in range(nt):
    # Nonlinear Term
    f = psi ** 3 - 3 * psi

    # Compute DCT of psi and f
    hat_psi = torch.tensor(dctn(psi.cpu().numpy(), norm='ortho'), device=device)
    hat_f = torch.tensor(dctn(f.cpu().numpy(), norm='ortho'), device=device)

    # Update Rule in Fourier Space
    numerator = hat_psi + (sig * torch.tensor(dctn(bar_psi.cpu().numpy(), norm='ortho'), device=device) - (xi2 + eta2) * hat_f) * dt
    denominator = 1 + (sig + 2 * (xi2 + eta2) + eps2 * (xi2 + eta2) ** 2) * dt
    hat_psi_new = numerator / denominator

    # Inverse DCT to get updated psi
    psi = torch.tensor(idctn(hat_psi_new.cpu().numpy(), norm='ortho'), device=device)

t_end = time.time()
print("Elapsed time:", t_end - t_start)
