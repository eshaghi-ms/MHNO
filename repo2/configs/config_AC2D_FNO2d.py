import numpy as np

# General Setting
gpu_number = 'cuda'
# gpu_number = 'cuda:1'
torch_seed = 0
numpy_seed = 0

# Network Parameters
nTrjTrain = 1800  # 180  # 1800  # 80  # 50
nTrjTest = 100
batch_size = 25#50
learning_rate = 0.0001#0.001
weight_decay = 1e-4
epochs = 500
# iterations = epochs * (nTrain // batch_size)
modes = 16#10#16
width = 32#16#32
width_q = 32#128#32
width_h = 0
n_layers = 4
n_layers_q = 2#4#2
n_layers_h = 0

# Discretization
s = 64
T_in = 1
T_out = 90  # 90  # 45  # 10
T_total = 91

# Training Setting
normalized = True
# training = True
# load_model = False
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
