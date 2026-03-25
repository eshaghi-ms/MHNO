import os
import csv
import json
import importlib
import torch
import inspect
import random
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from training import train_fno, train_fno_time
from torch.utils.data import DataLoader, random_split
from utilities import ImportDataset, count_params, LpLoss, ModelEvaluator
from post_processing import plot_loss_trend, plot_field_trajectory, make_video, save_vtk

from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import Subset
################################################################
# Problem Definition
################################################################
problem = 'AC2D'
# problem = 'CH2D'
# problem = 'CH2DNL'
# problem = 'SH2D'
# problem = 'PFC2D'
# problem = 'MBE2D'
# problem = 'Navier_1e3'
# problem = 'Navier_1e5'

# network_name = 'TNO2d'
network_name = 'FNO3d'
# network_name = 'FNO2d'

# problem = 'AC3D'
# problem = 'CH3D'
# network_name = 'TNO3d'

print(f"problem = {problem}")
print(f"network = {network_name}")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
cf = importlib.import_module(f"configs.config_{problem}_{network_name}")
network = getattr(importlib.import_module('networks'), network_name)
torch.manual_seed(cf.torch_seed)
np.random.seed(cf.numpy_seed)
torch.cuda.manual_seed_all(cf.torch_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device(cf.gpu_number if torch.cuda.is_available() else 'cpu')
print("Device: ", device)
################################################################
# Load data and data normalization
################################################################
model_dir = os.path.join(problem, 'models')
model_name = f'{network_name}_{problem}_S{cf.s}_T{cf.T_in}to{cf.T_out}_width{cf.width}_modes{cf.modes}_q{cf.width_q}_h{cf.width_h}_Ablation_0.pt'

window_len = cf.T_in + cf.T_out
total_time = cf.T_total
windows_per_traj = total_time - window_len + 1
nTrain = cf.nTrjTrain * windows_per_traj
nTest = cf.nTrjTest * windows_per_traj

print(f"model = {model_name}")
print(f"number of epoch = {cf.epochs}")
print(f"batch size = {cf.batch_size}")
print(f"nTrjTrain = {cf.nTrjTrain}")
print(f"nTrjTest = {cf.nTrjTest}")
print(f"nTrain = {nTrain}")
print(f"nTest = {nTest}")
print(f"learning_rate = {cf.learning_rate}")
print(f"n_layers = {cf.n_layers}")
print(f"width_q = {cf.width_q}")
print(f"width_h = {cf.width_h}")

model_path = os.path.join(model_dir, model_name)
os.makedirs(model_dir, exist_ok=True)

dataset = ImportDataset(
    cf.parent_dir, cf.matlab_dataset,
    cf.normalized, cf.T_in, cf.T_out,
    cf.nTrjTrain, cf.nTrjTest,
    use_sliding_window=True
)

# 2) Sanity-check shapes:
nTraj = dataset.data.shape[0]
assert nTraj == cf.nTrjTrain + cf.nTrjTest
assert total_time == dataset.data.shape[1]

# 3) Build “group” array so that each sliding-window sample i belongs to trajectory groups[i]
groups = np.repeat(np.arange(nTraj), windows_per_traj)
assert len(groups) == len(dataset)

# 4) Do a group-aware train/test split by trajectory
gss = GroupShuffleSplit(
    n_splits=1,
    train_size=cf.nTrjTrain,
    test_size=cf.nTrjTest,
    random_state=42
)
train_idx, test_idx = next(
    gss.split(X=np.zeros(len(dataset)), groups=groups)
)

train_dataset = Subset(dataset, train_idx)
test_dataset = Subset(dataset, test_idx)
print('Train dataset len = ', len(train_dataset))
print('Test dataset len = ', len(test_dataset))

normalizers = [dataset.normalizer_x, dataset.normalizer_y] if cf.normalized else None

train_loader = DataLoader(train_dataset, batch_size=cf.batch_size, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=cf.batch_size, shuffle=False)

################################################################
# Training
################################################################
if network_name in ['FNO2d', 'TNO2d']:
    model = network(cf.modes, cf.modes, cf.width, cf.width_q, cf.width_h, cf.T_in, cf.T_out, cf.n_layers, cf.n_layers_q, cf.n_layers_h).to(device)
elif network_name in ['FNO3d', 'TNO3d']:
    model = network(cf.modes, cf.modes, cf.modes, cf.width, cf.width_q, cf.width_h, cf.T_in, cf.T_out, cf.n_layers, cf.n_layers_q, cf.n_layers_h).to(device)
else:
    raise Exception("network_name is not correct")

print(count_params(model))  # Print parameter count

train_mse_log, train_l2_log, test_l2_log = [], [], []

# Attempt to load a checkpoint if available
if os.path.exists(model_path) and cf.load_model:
    print(f"Loading pre-trained model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = checkpoint['model']
    train_mse_log = checkpoint.get('train_mse_log', [])
    train_l2_log = checkpoint.get('train_l2_log', [])
    test_l2_log = checkpoint.get('test_l2_log', [])
else:
    print("No pre-trained model loaded. Initializing a new model.")

# Define optimizer, scheduler, and loss function
iterations = cf.epochs * (nTrain // cf.batch_size)
optimizer = torch.optim.Adam(model.parameters(), lr=cf.learning_rate, weight_decay=cf.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
myloss = LpLoss(size_average=False)

# Train the model
if cf.training:
    if network_name == 'FNO2d':
        model, train_l2_log, test_l2_log = (
            train_fno_time(model, myloss, cf.epochs, cf.batch_size, train_loader, test_loader,
                           optimizer, scheduler, cf.normalized, normalizers, device))
        train_mse_log = []
    else:
        model, train_mse_log, train_l2_log, test_l2_log = train_fno(
            model, myloss, cf.epochs, cf.batch_size, train_loader, test_loader,
            optimizer, scheduler, cf.normalized, normalizers, device)
    print(f"Saving model and logs to {model_path}")
    torch.save({
        'model': model,
        'train_mse_log': train_mse_log,
        'train_l2_log': train_l2_log,
        'test_l2_log': test_l2_log
    }, model_path)

# Plot training‐loss curves
# losses = [train_mse_log, train_l2_log, test_l2_log]
# labels = ['Train MSE', 'Train L2', 'Test L2']
losses = [train_l2_log, test_l2_log]
labels = ['Train L2', 'Test L2']
plot_loss_trend(losses, labels, problem)

results_dir = os.path.join(problem, 'results')
os.makedirs(results_dir, exist_ok=True)
loss_log_path = os.path.join(results_dir, f"losses_{model_name}")

print(f"Saving losses to {loss_log_path}")
torch.save({
    'train_l2_log': train_l2_log,
    'test_l2_log': test_l2_log,
}, loss_log_path)
################################################################
# Train dataset Evaluation
################################################################
evaluator = ModelEvaluator(
    model, dataset, train_dataset,
    cf.s, cf.T_in, cf.T_out, device,
    cf.normalized, normalizers,
    time_history=(network_name == 'FNO2d'),
    use_sliding_window=True
)

results = evaluator.evaluate(loss_fn=myloss)

# 1) Print overall trajectory statistics
traj_stats = results["trajectory"]
print("Overall Trajectory Statistics: (Train Dataset)")
print(f"  Average L2 Error: {traj_stats['average']:.5f}")
print(f"  Std Dev:          {traj_stats['std_dev']:.5f}")
print(f"  Min Error:        {traj_stats['min']['value']:.5f} at indices {traj_stats['min']['index']}")
print(f"  Max Error:        {traj_stats['max']['value']:.5f} at indices {traj_stats['max']['index']}")
print(f"  Mode:             {traj_stats['mode']['value']:.5f} appearing in")
print(f"                    {traj_stats['mode']['count']} times at indices {traj_stats['mode']['indices']}")
print()

################################################################
# Evaluation
################################################################
print("########################################################")
print("##################### Test Dataset #####################")
print("########################################################")
evaluator = ModelEvaluator(
    model, dataset, test_dataset,
    cf.s, cf.T_in, cf.T_out, device,
    cf.normalized, normalizers,
    time_history=(network_name == 'FNO2d'),
    use_sliding_window=True
)

results = evaluator.evaluate(loss_fn=myloss)

# 1) Print overall trajectory statistics
traj_stats = results["trajectory"]
print("Overall Trajectory Statistics:")
print(f"  Average L2 Error: {traj_stats['average']:.5f}")
print(f"  Std Dev:          {traj_stats['std_dev']:.5f}")
print(f"  Min Error:        {traj_stats['min']['value']:.5f} at indices {traj_stats['min']['index']}")
print(f"  Max Error:        {traj_stats['max']['value']:.5f} at indices {traj_stats['max']['index']}")
print(f"  Mode:             {traj_stats['mode']['value']:.5f} appearing in")
print(f"                    {traj_stats['mode']['count']} times at indices {traj_stats['mode']['indices']}")
print()

"""
# 2) Print per-window statistics
print("Per-Window Error Statistics:")
for idx, ws in sorted(results["per_window"].items()):
    print(
        f" Window {idx:2d}: "
        f"count={ws['count']:3d}, "
        f"avg={ws['average']:.5f}, "
        f"std={ws['std_dev']:.5f}, "
        f"min={ws['min']:.5f}, "
        f"max={ws['max']:.5f}"
    )

# 3) Plot per-window average error
window_idxs = sorted(results["per_window"].keys())
avg_errs = [results["per_window"][i]["average"] for i in window_idxs]

plt.figure()
plt.plot(window_idxs, avg_errs)
plt.xlabel("Window Index")
plt.ylabel("Average L2 Error")
plt.title("Per-Window Average L2 Error")
plt.show()

# 4) Histogram of trajectory L2 errors
traj_errors = evaluator.test_l2_set if hasattr(evaluator, 'test_l2_set') else None
if traj_errors is not None:
    plt.figure()
    plt.hist(traj_errors)
    plt.xlabel("Trajectory L2 Error")
    plt.ylabel("Count")
    plt.title("Histogram of Trajectory L2 Errors")
    plt.show()
"""
# 5) Plot mean ± std + min/max as before
window_idxs = sorted(results["per_window"].keys())
avg_errs = [results["per_window"][i]["average"] for i in window_idxs]
std_errs = [results["per_window"][i]["std_dev"] for i in window_idxs]
min_errs = [results["per_window"][i]["min"] for i in window_idxs]
max_errs = [results["per_window"][i]["max"] for i in window_idxs]

plt.figure(figsize=(5, 5))
plt.plot(window_idxs, avg_errs, color="C0", linewidth=2, label="Mean L2 Error")
lower = [m - s for m, s in zip(avg_errs, std_errs)]
upper = [m + s for m, s in zip(avg_errs, std_errs)]
plt.fill_between(window_idxs, lower, upper, color="C0", alpha=0.2, label="±1 Std Dev")
# plt.plot(window_idxs, min_errs, linestyle="--", color="C1", label="Min L2 Error")
# plt.plot(window_idxs, max_errs, linestyle="--", color="C2", label="Max L2 Error")
plt.xlabel("Window Index")
plt.ylabel("L2 Error")
plt.title("Per-Window Error Statistics")
plt.legend()
plt.ylim(bottom=0)
plt.grid(True)
plt.tight_layout()
plt.show()
"""
# Extract the raw tensors for later comparison
inp = results['input']
pred = results['prediction']
exact = results['exact']
################################################################
# Save out everything we need for later comparisons
################################################################
# Strip off the “.pt” suffix from model_name, then append our own suffixes
base_name = model_name[:-3]  # e.g. "FNO2d_AC2D_S64_T1to10_width40_modes14_q160_h0"

results_dir = os.path.join(problem, 'results')
os.makedirs(results_dir, exist_ok=True)

# 1) Save the entire `results` dictionary (includes all quartiles now)
results_path = os.path.join(results_dir, f'{base_name}_results.pt')
torch.save(results, results_path)
print(f"Saved full evaluation dictionary to: {results_path}")

# 2) Save the raw input/pred/exact as separate .pt files
input_path  = os.path.join(results_dir, f'{base_name}_input.pt')
pred_path   = os.path.join(results_dir, f'{base_name}_prediction.pt')
ground_path = os.path.join(results_dir, f'{base_name}_ground_truth.pt')

torch.save(inp,   input_path)
torch.save(pred,  pred_path)
torch.save(exact, ground_path)
print(f"Saved input tensor to:       {input_path}")
print(f"Saved prediction tensor to:   {pred_path}")
print(f"Saved ground-truth tensor to: {ground_path}")

# 3) Save per-window stats (with quartiles) to a CSV
csv_path = os.path.join(results_dir, f'{base_name}_per_window_stats.csv')
with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # Header now includes q1, median, q3
    writer.writerow([
        'window_idx',
        'count',
        'average',
        'std_dev',
        'min',
        'max',
        'q1',
        'median',
        'q3'
    ])
    for widx in window_idxs:
        ws = results['per_window'][widx]
        writer.writerow([
            widx,
            ws['count'],
            f"{ws['average']:.6f}",
            f"{ws['std_dev']:.6f}",
            f"{ws['min']:.6f}",
            f"{ws['max']:.6f}",
            f"{ws['q1']:.6f}",
            f"{ws['median']:.6f}",
            f"{ws['q3']:.6f}",
        ])
print(f"Saved per-window statistics CSV to: {csv_path}")

# 4) Save trajectory‐level stats (with quartiles) to a JSON
traj_stats_path = os.path.join(results_dir, f'{base_name}_trajectory_stats.json')
# We assume `traj_stats = results["trajectory"]` includes keys 'q1', 'median', 'q3'
with open(traj_stats_path, 'w') as f:
    json.dump(traj_stats, f, indent=4)
print(f"Saved trajectory statistics JSON to: {traj_stats_path}")
################################################################
# post-processing
################################################################
a_ind = inp[cf.index]
plot_field_trajectory(cf.domain, [a_ind], ['Initial Value'], [0], [cf.plot_range[0]], problem, colorbar=cf.colorbar)

u_pred = pred[cf.index]
u_exact = exact[cf.index]
error = torch.abs(u_pred-u_exact)
# error = u_pred-u_exact

# u_pred_e = torch.where(u_pred < -0.0, -1, torch.where(u_pred > 0.0, 1, 0))
# u_exact_e = torch.where(u_exact < -0.0, -1, torch.where(u_exact > 0.0, 1, 0))
# error_e = torch.abs(u_pred_e-u_exact_e)
# error = torch.where(error_e < 0.01, 0, torch.abs(u_pred-u_exact))

# error = torch.abs(u_pred-u_exact)

# Save as VTK files
# vtk_dir = os.path.join(problem, 'vtk_outputs')
# os.makedirs(vtk_dir, exist_ok=True)
# save_vtk(os.path.join(vtk_dir, 'u_pred.vti'), u_pred.cpu().numpy(), u_pred.cpu().numpy().shape)
# save_vtk(os.path.join(vtk_dir, 'u_exact.vti'), u_exact.cpu().numpy(), u_exact.cpu().numpy().shape)
# save_vtk(os.path.join(vtk_dir, 'error.vti'), error.cpu().numpy(), error.cpu().numpy().shape)

field_names = ['Exact Value', 'Predicted Value', 'Error']
fields = [u_exact, u_pred, error]
plot_field_trajectory(cf.domain, fields, field_names, cf.time_steps, cf.plot_range, problem, plot_show=False, colorbar=cf.colorbar)

# time_steps = [0, 19, 44, 69, 89]
# indices = random.sample(range(len(pred)), 20)
# for ind in indices:
#     field_names = [f"Exact Value_{ind}", f"Predicted Value_{ind}"]
#     u_pred = pred[ind]
#     u_exact = exact[ind]
#     fields = [u_exact, u_pred]
#     plot_field_trajectory(cf.domain, fields, field_names, time_steps, cf.plot_range, problem + "/random_plot", plot_show=False, colorbar=False)

# make_video(u_pred, cf.domain, "predicted", plot_range, problem)
# make_video(u_exact, cf.domain, "exact", plot_range, problem)

"""