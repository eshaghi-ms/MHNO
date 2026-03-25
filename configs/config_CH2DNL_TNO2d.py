"""
problem = CH2DNL
network = TNO2d
Device:  cuda:1
model = TNO2d_CH2DNL_S64_T1to90_width32_modes12_q64_h32.pt
number of epoch = 500
batch size = 25
nTrjTrain = 4200
nTrjTest = 200
nTrain = 4200
nTest = 200
learning_rate = 0.001
n_layers = 6
width_q = 64
width_h = 32
Found saved dataset at ./data/CH2DNL_4400_Nt_101_Nx_64.pt
Dataset (Train + Test) input shape =  torch.Size([4400, 64, 64, 1])
Dataset (Train + Test) output shape =  torch.Size([4400, 64, 64, 90])
Train dataset len =  4200
Test dataset len =  200
3762643

498 20.974560245998873 0.0001904517232885285 0.03346981861761638 0.07166961014270783
499 20.886749355002394 0.00019044512642029163 0.03346922471409752 0.07166924595832824

-------------------------------------------------------------
problem = CH2DNL
network = TNO2d
Device:  cuda:3
model = TNO2d_CH2DNL_S64_T1to45_width32_modes12_q64_h32.pt
number of epoch = 500
batch size = 25
nTrjTrain = 120
nTrjTest = 200
nTrain = 5520
nTest = 9200
learning_rate = 0.001
n_layers = 6
width_q = 64
width_h = 32
Found saved dataset at ./data/CH2DNL_4400_Nt_101_Nx_64.pt
Dataset (Train + Test) input shape =  torch.Size([14720, 64, 64, 1])
Dataset (Train + Test) output shape =  torch.Size([14720, 64, 64, 45])
Train dataset len =  5520
Test dataset len =  9200
3660313

498 27.34172726700126 1.4068234970297178e-05 0.009020705307847347 0.12348005929718847
499 27.515579246002744 1.4069906275868743e-05 0.009021080719927946 0.12347881366377292

Overall Trajectory Statistics:
  Average L2 Error: 0.22380
  Std Dev:          0.04168
  Min Error:        0.13562
  Max Error:        0.34885

--------------------------------------------------------------------------------------

problem = CH2DNL
network = TNO2d
Device:  cuda:3
model = TNO2d_CH2DNL_S64_T1to90_width40_modes18_q40_h32.pt
number of epoch = 500
batch size = 20
nTrjTrain = 4200
nTrjTest = 100
nTrain = 4200
nTest = 100
learning_rate = 0.001
n_layers = 6
width_q = 40
width_h = 32
Found saved dataset at ./data/CH2DNL_4400_Nt_101_Nx_64.pt
Dataset (Train + Test) input shape =  torch.Size([4300, 64, 64, 1])
Dataset (Train + Test) output shape =  torch.Size([4300, 64, 64, 90])
Train dataset len =  4200
Test dataset len =  100
12819171

498 31.663059478989453 6.131372013312232e-05 0.019024386129208974 0.04993417739868164
499 31.315211928013014 6.131071594219455e-05 0.019023911768481845 0.04993555784225464

Overall Trajectory Statistics:
  Average L2 Error: 0.04994
  Std Dev:          0.01254
  Min Error:        0.03101
  Max Error:        0.09603

"""

import numpy as np

# General Setting
gpu_number = 'cuda:3'
# gpu_number = 'cuda'
torch_seed = 0
numpy_seed = 0

# Network Parameters
nTrjTrain = 4200  # 4200  # 120  # 80
nTrjTest = 100
batch_size = 20#25  # 25  # 50
learning_rate = 0.0005
weight_decay = 1e-4
epochs = 500
# iterations = epochs * (nTrain // batch_size)
modes = 20#16  # 18
width = 40#36#32  # 42
width_q = 40#36#32  # 42
width_h = 40#32#42
n_layers = 6#4#6  # 4
n_layers_q = 2
n_layers_h = 6#4#2

# Discretization
s = 64
T_in = 1
T_out = 90  # 90  # 45  # 10
T_total = 91

# Training Setting
normalized = True
training = True
load_model = False
# training = False
# load_model = True

# Database
parent_dir = './data/'
matlab_dataset = 'CH2DNL_4400_Nt_101_Nx_64.mat'

# Plotting
index = 10  # 24  # 236  # 197  # 24
domain = [-3.0, 3.0]
time_steps = [9, 29, 49, 69, 89, 99]
plot_range = [[-0.5, 0.5], [-0.5, 0.5], [-0.0, 1.0]]
colorbar = False
