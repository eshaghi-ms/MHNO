import numpy as np

# General Setting
gpu_number = 'cuda:2'
# gpu_number = 'cuda'
torch_seed = 0
numpy_seed = 0

# Network Parameters
nTrjTrain = 80  # 4200  # 120  # 80
nTrjTest = 200
batch_size = 25
learning_rate = 0.005
weight_decay = 1e-4
epochs = 500
# iterations = epochs * (nTrain // batch_size)
modes = 8
width = 16  # 10  # 8
width_q = 16  # 8  # 4
width_h = 8  # 8  # 4
n_layers = 4
n_layers_q = 2
n_layers_h = 0

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
matlab_dataset = 'MBE2D_4440_Nt_101_Nx_64.mat'

# Plotting
index = 386  # 24  # 62  3  244
domain = [-np.pi, np.pi]
time_steps = [9, 29, 49, 69, 89, 99]
plot_range = [[-0.5, 0.5], [-0.5, 0.5], [-0.0, 1.0]]
colorbar = False
