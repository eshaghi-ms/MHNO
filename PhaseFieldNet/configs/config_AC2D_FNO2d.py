import numpy as np

# General Setting
gpu_number = 'cuda:3'
torch_seed = 0
numpy_seed = 0

# Network Parameters
nTrain = 1000
nTest = 100
batch_size = 20  # 50  # 100
learning_rate = 0.001
weight_decay = 1e-4
epochs = 1000  # 200  # 100
iterations = epochs * (nTrain // batch_size)
modes = 14  # 14  # 12
width = 40  # 32  # 32
width_q = 160  # 128  # 128
width_h = 0
n_layers = 4
n_layers_q = 2
n_layers_h = 0

# Discretization
s = 64
T_in = 1
T_out = 100  # 50  # 10

# Training Setting
normalized = True
training = False
load_model = True

# Database
parent_dir = './data/'
matlab_dataset = 'AC2D_2000_Nt_101_Nx_64.mat'

# Plotting
index = 57
domain = [-np.pi, np.pi]
time_steps = [9, 29, 49, 69, 89, 99]
plot_range = [[-0.5, 0.5], [-0.5, 0.5], [-0.0, 1.0]]
colorbar = False
