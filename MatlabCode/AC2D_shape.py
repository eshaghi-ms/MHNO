import time

import torch
import numpy as np
import matplotlib.pyplot as plt
from GRF import GRF
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

# Parameter Initialization

# Spatial Parameters
Nx = 256
Ny = 256
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

# Initial Condition
ICs = {
    "Random",
    # "Circle",
    # "Ellipse",
    # "Square",
    # "Ring",
    # "Heart",
    # "Star",
}

# Move to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for IC in ICs:
    # Time Discretization
    dt = 0.1
    T = 10
    Nt = round(T / dt)
    ns = 1

    if IC == "Random":
        dt = 0.001
        T = 0.1
        Nt = round(T / dt)
        ns = 1
        np.random.seed(3)
        tau = 100
        alpha = 10
        h = 20
        norm_a = GRF(alpha, tau, Nx-2*h)
        norm_a = norm_a - 0.2 * np.std(norm_a)
        # norm_a = norm_a * (xx.numpy() - np.pi) * (xx.numpy() + np.pi) * (yy.numpy() - np.pi) * (yy.numpy() + np.pi)
        # Create the u matrix with -1 and 1 values based on GRF outcome
        u = -np.ones((Nx, Ny))
        u[h:-h, h:-h][norm_a >= 0] = 1
        u[h:-h, h:-h][norm_a < 0] = -1
        u = torch.tensor(u)
    elif IC == "Circle":
        dt = 1
        T = 450
        Nt = round(T / dt)
        # Initial condition: circular pattern
        u = torch.full((Nx, Ny), -1.0)
        # Coordinates normalized to center
        X, Y = xx, yy
        radius = 2.0  # set the radius of the circle
        # Create a circle in the center
        mask = (X ** 2 + Y ** 2) < radius ** 2
        u[mask] = 1.0
    elif IC == "Ellipse":
        dt = 1
        T = 200
        Nt = round(T / dt)
        u = torch.full((Nx, Ny), -1.0)
        a, b = 2.0, 1.0  # major and minor axes
        mask = ((xx / a) ** 2 + (yy / b) ** 2) < 1.0
        u[mask] = 1.0
    elif IC == "Square":
        dt = 1
        T = 150
        Nt = round(T / dt)
        u = torch.full((Nx, Ny), -1.0)
        half_width = 1.5
        mask = (xx.abs() < half_width) & (yy.abs() < half_width)
        u[mask] = 1.0
    elif IC == "Ring":
        dt = 0.1
        T = 10
        Nt = round(T / dt)
        u = torch.full((Nx, Ny), -1.0)
        r = torch.sqrt(xx ** 2 + yy ** 2)
        mask = (r > 1.0) & (r < 2.0)
        u[mask] = 1.0
    elif IC == "Heart":
        u = torch.full((Nx, Ny), -1.0)
        X = xx / np.pi * 2
        Y = yy / np.pi * 2
        mask = ((X ** 2 + Y ** 2 - 1) ** 3 - 2 * X ** 2 * Y ** 3) < 0
        u[mask] = 1.0
    elif IC == "Star":
        r = torch.sqrt(xx ** 2 + yy ** 2)
        theta = torch.atan2(yy, xx)
        star = 1.5 + 0.9 * torch.sin(5 * theta)
        u = torch.full((Nx, Ny), -1.0)
        u[r < star] = 1.0

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


    def plot_snap(u, title=None):
        colors = [(1, 1, 1), (73/255, 67/255, 89/255)]  # white to black
        cmap = LinearSegmentedColormap.from_list('white_black', colors, N=256)

        colors = [(1, 1, 1), (0.1, 0.1, 0.5)]  # white to black
        cmap = LinearSegmentedColormap.from_list('white_black', colors, N=256)

        plt.figure(figsize=(6, 6))
        plt.contourf(xx.numpy(), yy.numpy(), u, levels=[0.0, 1.1], cmap=cmap)
        plt.axis('off')  # hide axis ticks and labels
        plt.tight_layout()
        if title is not None:
            plt.savefig(title + ".png", dpi=900)
        plt.show()


    # Move the result back to CPU for plotting
    # for i in range(10):
    #     plot_snap(U[i*10].cpu().numpy())
    # plot_snap(U[0].cpu().numpy())
    # plot_snap(U[2].cpu().numpy(), f"AC_{IC}_2")
    plot_snap(U[10].cpu().numpy(), f"AC_{IC}_10")
    plot_snap(U[50].cpu().numpy(), f"AC_{IC}_55")
    plot_snap(U[-1].cpu().numpy(), f"AC_{IC}_100")
