import numpy as np

# General Setting
gpu_number = 'cuda:0'
# gpu_number = 'cuda'
torch_seed = 0
numpy_seed = 0

# Network Parameters
nTrjTrain = 80  # 4200  # 120  # 80
nTrjTest = 200
batch_size = 25
learning_rate = 0.001
weight_decay = 1e-4
epochs = 500
# iterations = epochs * (nTrain // batch_size)
modes = 12
width = 32
width_q = 64
width_h = 32
n_layers = 6
n_layers_q = 2
n_layers_h = 2

# Discretization
s = 64
T_in = 1
T_out = 10  # 90  # 45  # 10
T_total = 91

# Training Setting
normalized = True
# training = True
# load_model = False
training = False
load_model = True

# Database
parent_dir = './data/'
matlab_dataset = 'CH2D_4400_Nt_101_Nx_64.mat'

# Plotting
index = 12  # 191 # 12
domain = [-0.5, 0.5]
time_steps = [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89]
plot_range = [[-0.5, 0.5], [-0.5, 0.5], [-0.0, 1.0]]
colorbar = True
