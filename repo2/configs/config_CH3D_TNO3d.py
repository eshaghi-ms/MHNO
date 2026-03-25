import numpy as np

# General Setting
gpu_number = 'cuda:3'
torch_seed = 0
numpy_seed = 0

# Network Parameters
nTrain = 2000
nTest = 200
batch_size = 50
learning_rate = 0.001
weight_decay = 1e-4
epochs = 1000
iterations = epochs * (nTrain // batch_size)
modes = 8
width = 32
width_q = 2 * width
width_h = width
n_layers = 4

# Discretization
s = 32
T_in = 1
T_out = 100

# Training Setting
normalized = True
training = True  # False
load_model = False  # True

# Database
parent_dir = './data/'
matlab_dataset = 'CH3D_2200_Nt_101_Nx_32.mat'

# Plotting
index = 12
domain = [-np.pi, np.pi]
# time_steps = [29, 69]
time_steps = [39, 49, 59, 69, 79, 89, 99]
# time_steps = [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49,
#               54, 59, 64, 69, 74, 79, 84, 89, 94, 99]
