import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def get_grid_2d(shape, device):
    batchsize, size_x, size_y = shape[0], shape[1], shape[2]
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
    gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
    return torch.cat((gridx, gridy), dim=-1).to(device)


def get_grid_3d(shape, device):
    batchsize, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3]
    gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
    gridx = gridx.reshape(1, size_x, 1, 1, 1).repeat([batchsize, 1, size_y, size_z, 1])
    gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
    gridy = gridy.reshape(1, 1, size_y, 1, 1).repeat([batchsize, size_x, 1, size_z, 1])
    gridz = torch.tensor(np.linspace(0, 1, size_z), dtype=torch.float)
    gridz = gridz.reshape(1, 1, 1, size_z, 1).repeat([batchsize, size_x, size_y, 1, 1])
    return torch.cat((gridx, gridy, gridz), dim=-1).to(device)


# Complex multiplication
def compl_mul2d(inp, weights):
    # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
    return torch.einsum("bixy,ioxy->boxy", inp, weights)


def compl_mul3d(inp, weights):
    # (batch, in_channel, x,y,t ), (in_channel, out_channel, x,y,t) -> (batch, out_channel, x,y,t)
    return torch.einsum("bixyz,ioxyz->boxyz", inp, weights)


class SpectralConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coefficients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x


class SpectralConv3d(nn.Module):

    def __init__(self, in_channels, out_channels, modes1, modes2, modes3):
        super(SpectralConv3d, self).__init__()

        """
        3D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2
        self.modes3 = modes3

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights2 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights3 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))
        self.weights4 = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, self.modes3,
                                    dtype=torch.cfloat))

    def forward(self, x):
        batchsize = x.shape[0]
        # Compute Fourier coefficients up to factor of e^(- something constant)
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2, :self.modes3] = \
            compl_mul3d(x_ft[:, :, :self.modes1, :self.modes2, :self.modes3], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2, :self.modes3] = \
            compl_mul3d(x_ft[:, :, -self.modes1:, :self.modes2, :self.modes3], self.weights2)
        out_ft[:, :, :self.modes1, -self.modes2:, :self.modes3] = \
            compl_mul3d(x_ft[:, :, :self.modes1, -self.modes2:, :self.modes3], self.weights3)
        out_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3] = \
            compl_mul3d(x_ft[:, :, -self.modes1:, -self.modes2:, :self.modes3], self.weights4)

        # Return to physical space
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class MLP2d(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, T=1, num_layers=2):
        """
        Initialize the MLP2d class.
        Parameters:
        - in_channels: Number of input channels.
        - out_channels: Number of output channels.
        - mid_channels: Number of intermediate channels.
        - T: Number of network blocks (default=1).
        - num_layers: Number of layers in each block (default=2).
        """
        super(MLP2d, self).__init__()

        self.num_layers = num_layers
        self.layers = nn.ModuleList()

        for _ in range(T):
            self.layers.append(nn.Conv2d(in_channels, mid_channels, 1))
            for _ in range(self.num_layers - 2):
                self.layers.append(nn.Conv2d(mid_channels, mid_channels, 1))
            self.layers.append(nn.Conv2d(mid_channels, out_channels, 1))

    def forward(self, x, t=0):
        start = t * self.num_layers
        end = start + self.num_layers
        for i in range(start, end - 1):
            x = F.gelu(self.layers[i](x))
        x = self.layers[end - 1](x)
        return x


class MLP3d(MLP2d):
    def __init__(self, in_channels, out_channels, mid_channels, T=1, num_layers=2):
        super(MLP3d, self).__init__(in_channels, out_channels, mid_channels, T, num_layers)

        self.layers = nn.ModuleList()
        for _ in range(T):
            self.layers.append(nn.Conv3d(in_channels, mid_channels, 1))
            for _ in range(self.num_layers - 2):
                self.layers.append(nn.Conv3d(mid_channels, mid_channels, 1))
            self.layers.append(nn.Conv3d(mid_channels, out_channels, 1))


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, width_q, T_in, T_out, n_layers, n_layers_q):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the previous 10 time steps + 2 locations (u(t-10, x, y), ..., u(t-1, x, y),  x, y)
        input shape: (batchsize, x=64, y=64, c=12)
        output: the solution of the next timestep
        output shape: (batchsize, x=64, y=64, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.width_q = width_q
        self.T_in = T_in
        self.T_out = T_out
        self.padding = 8  # pad the domain if input is non-periodic
        self.n_layers = n_layers
        self.n_layers_q = n_layers_q

        self.p = nn.Linear(T_in + 2, self.width)
        self.convs = nn.ModuleList(
            [SpectralConv2d(self.width, self.width, self.modes1, self.modes2) for _ in range(n_layers)])
        self.mlps = nn.ModuleList([MLP2d(self.width, self.width, self.width) for _ in range(n_layers)])
        self.ws = nn.ModuleList([nn.Conv2d(self.width, self.width, 1) for _ in range(n_layers)])
        self.norm = nn.InstanceNorm2d(self.width)
        self.q = MLP2d(self.width, 1, self.width_q, num_layers=self.n_layers_q)  # output channel is 1: u(x, y)

    def forward(self, x):
        grid = get_grid_2d(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0,self.padding, 0,self.padding]) # pad the domain if input is non-periodic

        for i in range(self.n_layers):
            x1 = self.convs[i](x)
            x1 = self.mlps[i](x1)
            x2 = self.ws[i](x)
            x = x1 + x2
            x = F.gelu(x) if i < self.n_layers - 1 else x

        # x = x[..., :-self.padding, :-self.padding] # pad the domain if input is non-periodic
        x = self.q(x)
        x = x.permute(0, 2, 3, 1)
        return x


class TNO2d(FNO2d):
    def __init__(self, modes1, modes2, width, width_q, width_h, T_in, T_out, n_layers, n_layers_q, n_layers_h):
        super(TNO2d, self).__init__(modes1, modes2, width, width_q, T_in, T_out, n_layers, n_layers_q)

        self.width_h = width_h
        self.n_layers_h = n_layers_h
        self.q = MLP2d(self.width, 1, self.width_q, T_out, self.n_layers_q)
        self.h = MLP2d(1, 1, self.width_h, T_out - 1, self.n_layers_h)

    def forward(self, x):
        grid = get_grid_2d(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)
        # x = F.pad(x, [0, self.padding, 0, self.padding])

        for i in range(self.n_layers):
            x1 = self.convs[i](x)
            x1 = self.mlps[i](x1)
            x2 = self.ws[i](x)
            x = x1 + x2
            x = F.gelu(x) if i < self.n_layers - 1 else x

        # x = x[..., :-self.padding, :-self.padding]
        X = torch.zeros(*grid.shape[:-1], self.T_out, device=x.device)
        # Q = torch.zeros(*grid.shape[:-1], self.T_out, device=x.device) # temp
        # H = torch.zeros(*grid.shape[:-1], self.T_out, device=x.device) # temp
        xt = self.q(x)
        X[..., 0] = xt.permute(0, 2, 3, 1).squeeze(-1)
        # Q[..., 0] = xt.permute(0, 2, 3, 1).squeeze(-1) # temp

        for t in range(1, self.T_out):
            x1 = self.q(x, t)
            x2 = self.h(xt, t - 1)

            xt = x1 + x2
            X[..., t] = xt.permute(0, 2, 3, 1).squeeze(-1)
            # Q[..., t] = x1.permute(0, 2, 3, 1).squeeze(-1)
            # H[..., t] = x2.permute(0, 2, 3, 1).squeeze(-1)
        # return X, Q, H  # temp
        return X


class FNO3d(nn.Module):
    def __init__(self, modes1, modes2, modes3, width, width_q, T_in, T_out, n_layers, n_layers_q):
        super(FNO3d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .

        input: the solution of the first 10 time_steps + 3 locations (u(1, x, y), ..., u(10, x, y),  x, y, t). It's a constant function in time, except for the last index.
        input shape: (batchsize, x=64, y=64, t=40, c=13)
        output: the solution of the next 40 time_steps
        output shape: (batchsize, x=64, y=64, t=40, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.modes3 = modes3
        self.width = width
        self.width_q = width_q
        self.T_in = T_in
        self.T_out = T_out
        self.padding = 6  # pad the domain if input is non-periodic
        self.n_layers = n_layers
        self.n_layers_q = n_layers_q

        self.p = nn.Linear(self.T_in + 3, self.width)

        self.convs = nn.ModuleList(
            [SpectralConv3d(self.width, self.width, self.modes1, self.modes2, self.modes3) for _ in range(n_layers)])
        self.mlps = nn.ModuleList([MLP3d(self.width, self.width, self.width) for _ in range(n_layers)])
        self.ws = nn.ModuleList([nn.Conv3d(self.width, self.width, 1) for _ in range(n_layers)])
        self.q = MLP3d(self.width, 1, self.width_q, num_layers=self.n_layers_q)

    def forward(self, x):
        x = x.unsqueeze(3).repeat([1, 1, 1, self.T_out, 1])
        grid = get_grid_3d(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 4, 1, 2, 3)
        x = F.pad(x, [0, self.padding])  # pad the domain if input is non-periodic

        for i in range(self.n_layers):
            x1 = self.convs[i](x)
            x1 = self.mlps[i](x1)
            x2 = self.ws[i](x)
            x = x1 + x2
            x = F.gelu(x) if i < self.n_layers - 1 else x

        x = x[..., :-self.padding]
        x = self.q(x)

        x = x.permute(0, 2, 3, 4, 1)[..., 0]  # pad the domain if input is non-periodic
        return x


class TNO3d(FNO3d):
    def __init__(self, modes1, modes2, modes3, width, width_q, width_h, T_in, T_out, n_layers, n_layers_q, n_layers_h):
        super(TNO3d, self).__init__(modes1, modes2, modes3, width, width_q, T_in, T_out, n_layers, n_layers_q)
        """
        input: the initial condition and locations (a(x, y, z), x, y, z)
        input shape: (batchsize, x=s, y=s, z=s, c=4)
        output: the solution 
        output shape: (batchsize, x=s, y=s, z=s, t=T)
        """
        self.width_h = width_h
        self.n_layers_h = n_layers_h

        #self.q = MLP3d(self.width, 1, self.width, T_out)
        #self.q2 = MLP3d(1, 1, self.width // 4, T_out - 1)
        self.q = MLP3d(self.width, 1, self.width_q, T_out, self.n_layers_q)
        self.h = MLP3d(1, 1, self.width_h, T_out - 1, self.n_layers_q)

    def forward(self, x):
        grid = get_grid_3d(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 4, 1, 2, 3)
        # x = F.pad(x, [0, self.padding, 0, self.padding])

        for i in range(self.n_layers):
            x1 = self.convs[i](x)
            x1 = self.mlps[i](x1)
            x2 = self.ws[i](x)
            x = x1 + x2
            x = F.gelu(x) if i < self.n_layers - 1 else x

        # x = x[..., :-self.padding, :-self.padding]
        X = torch.zeros(*grid.shape[:-1], self.T_out, device=x.device)
        xt = self.q(x)
        X[..., 0] = xt.permute(0, 2, 3, 4, 1).squeeze(-1)
        for t in range(1, self.T_out):
            x1 = self.q(x, t)
            x2 = self.h(xt, t - 1)
            xt = x1 + x2
            X[..., t] = xt.permute(0, 2, 3, 4, 1).squeeze(-1)
        return X
