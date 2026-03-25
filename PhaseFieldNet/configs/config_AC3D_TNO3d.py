import numpy as np

# General Setting
gpu_number = 'cuda:2'
torch_seed = 0
numpy_seed = 0

# Network Parameters
nTrain = 1000
nTest = 100
batch_size = 25
learning_rate = 0.001
weight_decay = 1e-4
epochs = 900  # 100
iterations = epochs * (nTrain // batch_size)
modes = 8
width = 32
width_q = width
width_h = width // 4
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
matlab_dataset = 'AC3D_1200_Nt_101_Nx_32.mat'

# Plotting
index = 12
domain = [-np.pi, np.pi]
# time_steps = [29, 69]
time_steps = [39, 49, 59, 69, 79, 89, 99]
# time_steps = [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49,
#               54, 59, 64, 69, 74, 79, 84, 89, 94, 99]
