import numpy as np

# General Setting
gpu_number = 'cuda:1'  # 'cuda:1'
torch_seed = 0
numpy_seed = 0

# Network Parameters
nTrain = 4000
nTest = 400
batch_size = 10  # 50  # 100
learning_rate = 0.001
weight_decay = 1e-4
epochs = 1000  # 200  # 100
iterations = epochs * (nTrain // batch_size)
modes = 18  # 16  # 12
width = 42  # 32  # 16
width_q = 64  # 32  # 16
width_h = 40  # 32  # 16
n_layers = 6  # 4  # 4
n_layers_q = 2
n_layers_h = 0

# Discretization
s = 64
T_in = 1
T_out = 100  # 50  # 10# 80

# Training Setting
normalized = True
training = False
load_model = True

# Database
parent_dir = './data/'
matlab_dataset = 'CH2DNL_4400_Nt_101_Nx_64.mat'

# Plotting
index = 62  # 24 # 62
domain = [-3.0, 3.0]
time_steps = [9, 29, 49, 69, 89, 99]
plot_range = [[-0.5, 0.5], [-0.5, 0.5], [-0.0, 1.0]]
colorbar = False
