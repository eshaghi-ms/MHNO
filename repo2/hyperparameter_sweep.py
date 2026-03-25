"""
hyperparam_sweep.py

This script performs a hyperparameter sweep to assess the influence of various hyperparameters on
the performance of a neural operator model (e.g., FNO2d or TNO2d) for a given problem (e.g., AC2D).
It systematically varies:
    - nTrjTrain
    - nTrjTest
    - batch_size
    - learning_rate
    - weight_decay
    - epochs
    - modes
    - width
    - width_q
    - width_h
    - n_layers
    - n_layers_q
    - n_layers_h
    - T_out

For each combination, it:
    1. Updates a copy of the base configuration (configs/config_<problem>_<network>.py).
    2. Loads the data, builds model, trains, and evaluates.
    3. Records the final test L2-error (or other metric) in a CSV file.

Usage:
    python hyperparam_sweep.py

Before running, ensure that:
    - `main.py` and its dependencies are in the same directory or on PYTHONPATH.
    - The base config for your problem/network (e.g., configs/config_AC2D_FNO2d.py) is present.
    - Adjust `PROBLEM`, `NETWORK`, and `BASE_CONFIG` at the top of this script if needed.
"""

import os
import csv
import sys
import json
import types
import torch
import random
import argparse
import itertools
import importlib
import numpy as np
from datetime import datetime
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Subset

# Make sure the repository root (where main.py resides) is on the PYTHONPATH
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

# Import utilities from the repo
from utilities import ImportDataset, count_params, LpLoss, ModelEvaluator
from training import train_fno, train_fno_time

# ------------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------------
SEED = 42  # pick any integer you like
os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # needed for full cuBLAS determinism

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False  # turn off data‑dependent autotune
torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int):
    # Each worker gets a different, but reproducible, seed.
    worker_seed = SEED + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g_loader = torch.Generator()
g_loader.manual_seed(SEED)

# =============================================================================
# USER-DEFINED SETTINGS
# =============================================================================
# Problem and network identifiers: must correspond to existing config_<problem>_<network>.py
parser = argparse.ArgumentParser(
    description="Run a hyper‑parameter sweep for FNO/TNO models."
)
parser.add_argument("--problem", default="AC2D", help="Problem identifier")
parser.add_argument("--network", default="FNO2d", help="Network identifier")
parser.add_argument("--param", default="lr", help="Primary swept parameter")
parser.add_argument("--gpu", default="cuda:1", help="GPU id or 'cpu'")
parser.add_argument("--t_out", type=int, default=10, help="Prediction horizon")

args = parser.parse_args()

PROBLEM, NETWORK, PARAM = args.problem, args.network, args.param
GPU, T_OUT = args.gpu, args.t_out
BASE_CONFIG = f'configs.config_{PROBLEM}_{NETWORK}'

# Directory where results (CSV, logs) will be saved
OUTPUT_DIR = os.path.join(REPO_ROOT, PROBLEM, "hyperparam_sweep_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Base configuration module (will be imported and then copied/modified)
base_cf_module = importlib.import_module(BASE_CONFIG)
base_cf_module.gpu_number = GPU

# Device selection
DEVICE = torch.device(base_cf_module.gpu_number if torch.cuda.is_available() else 'cpu')


# =============================================================================
# DEFINE HYPERPARAMETER GRID
# =============================================================================
def load_grid_override(file_path: str) -> dict:
    """Load JSON or YAML hyperparameter grid from a required file path."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Hyperparameter grid file not found: {file_path}")
    with open(file_path, "r", encoding="utf-8") as f:
        if file_path.endswith((".yml", ".yaml")):
            import yaml
            return yaml.safe_load(f)
        return json.load(f)


HYPERPARAM_Path = os.path.join(REPO_ROOT, "configs", "grids", f"grid_{PROBLEM}_{NETWORK}_Tout{T_OUT}_{PARAM}.json")
HYPERPARAM_GRID = load_grid_override(HYPERPARAM_Path)

# # -------------------------------------------------------------------------
# # Copy base config
# # -------------------------------------------------------------------------
# # We import the base config module (e.g., configs.config_AC2D_FNO2d)
# # cf = copy.deepcopy(base_cf_module)
# cf = types.SimpleNamespace(**{
#     k: v for k, v in vars(base_cf_module).items()
#     if not (k.startswith("__"))
# })
#
# cf.nTrjTrain = cf.nTrjTrain // 10
# cf.window_len = cf.T_in + cf.T_out
# cf.total_time = cf.T_total
# cf.windows_per_traj = cf.T_total - cf.window_len + 1
# cf.nTrain = cf.nTrjTrain * cf.windows_per_traj
# cf.nTest = cf.nTrjTest * cf.windows_per_traj

# ----------------------------------------------------------------------
# GLOBAL CACHES  (add these near the top of your file, e.g. after imports)
# ----------------------------------------------------------------------
_train_dataset = None
_test_dataset = None
_train_loader = None
_test_loader = None
_normalizers = None
_dataset = None


# =============================================================================
# FUNCTION: run_single_experiment
# =============================================================================
def run_single_experiment(param_dict, run_id):
    """
    Given a dictionary of hyperparameters (param_dict), this function:
      1. Creates a (deep) copy of the base config module.
      2. Updates the copied config with the new hyperparameters.
      3. Performs data loading, model creation, training, and evaluation.
      4. Returns a dictionary of results, including final test L2-error and any logs.

    Args:
        param_dict (dict): keys correspond to attributes in the config module to override.
        run_id (int): unique identifier for this particular run (for logging/filenames).

    Returns:
        results (dict): {
            'run_id': int,
            'hyperparams': param_dict,
            'n_params': int,           # number of model parameters
            'final_test_L2': float,    # final test L2-error
            'train_time_sec': float,   # (optional) total training time
        }
    """
    global _train_dataset, _test_dataset, _train_loader, _test_loader, _normalizers, _dataset
    # -------------------------------------------------------------------------
    # 1) override hyperparameters
    # -------------------------------------------------------------------------
    # We import the base config module (e.g., configs.config_AC2D_FNO2d)
    # cf = copy.deepcopy(base_cf_module)
    cf = types.SimpleNamespace(**{
        k: v for k, v in vars(base_cf_module).items()
        if not (k.startswith("__"))
    })
    # Override hyperparameters in the config object
    for key, val in param_dict.items():
        setattr(cf, key, val)

    cf.training = True
    cf.load_model = False

    # Recompute dependent quantities in main.py that rely on config values:
    #   - window_len, total_time, windows_per_traj, nTrain, nTest
    cf.window_len = cf.T_in + cf.T_out
    cf.total_time = cf.T_total
    cf.windows_per_traj = cf.T_total - cf.window_len + 1
    cf.nTrain = cf.nTrjTrain * cf.windows_per_traj
    cf.nTest = cf.nTrjTest * cf.windows_per_traj

    # -------------------------------------------------------------------------
    # 2) Print / Log current hyperparameter setting
    # -------------------------------------------------------------------------
    print(f"\n=== Run ID {run_id} ===")
    print("Hyperparameters:")
    for k, v in param_dict.items():
        print(f"  {k} = {v}")
    print(f"Derived: nTrain = {cf.nTrain}, nTest = {cf.nTest}, windows_per_traj = {cf.windows_per_traj}")

    # -------------------------------------------------------------------------
    # 3) Load dataset and build DataLoaders
    # -------------------------------------------------------------------------
    first_call = _train_dataset is None
    if first_call:
        _dataset = ImportDataset(
            cf.parent_dir, cf.matlab_dataset,
            cf.normalized, cf.T_in, cf.T_out,
            cf.nTrjTrain, cf.nTrjTest,
            use_sliding_window=True
        )

        nTraj = _dataset.data.shape[0]
        assert nTraj == cf.nTrjTrain + cf.nTrjTest, (
            f"Mismatch: dataset has {nTraj} trajectories, "
            f"but config requests {cf.nTrjTrain + cf.nTrjTest}"
        )

        groups = np.repeat(np.arange(nTraj), cf.windows_per_traj)
        gss = GroupShuffleSplit(
            n_splits=1,
            train_size=cf.nTrjTrain,
            test_size=cf.nTrjTest,
            random_state=42,
        )
        train_idx, test_idx = next(gss.split(X=np.zeros(len(_dataset)), groups=groups))
        _train_dataset = Subset(_dataset, train_idx)
        _test_dataset = Subset(_dataset, test_idx)

        _normalizers = [_dataset.normalizer_x, _dataset.normalizer_y] if cf.normalized else None

        _train_loader = DataLoader(
            _train_dataset,
            batch_size=cf.batch_size,
            shuffle=True,
            worker_init_fn=seed_worker,
            generator=g_loader,
        )
        _test_loader = DataLoader(
            _test_dataset,
            batch_size=cf.batch_size,
            shuffle=False,
            worker_init_fn=seed_worker,
            generator=g_loader,
        )

    # Re‑use the cached objects on every call
    # train_dataset = _train_dataset
    test_dataset = _test_dataset
    train_loader = _train_loader
    test_loader = _test_loader
    normalizers = _normalizers
    dataset = _dataset

    # -------------------------------------------------------------------------
    # 4) Build model
    # -------------------------------------------------------------------------
    if NETWORK in ['FNO2d', 'TNO2d']:
        model = getattr(importlib.import_module('networks'), NETWORK)(
            cf.modes, cf.modes, cf.width, cf.width_q, cf.width_h,
            cf.T_in, cf.T_out, cf.n_layers, cf.n_layers_q, cf.n_layers_h
        ).to(DEVICE)

    elif NETWORK in ['FNO3d', 'TNO3d']:
        model = getattr(importlib.import_module('networks'), NETWORK)(
            cf.modes, cf.modes, cf.modes,
            cf.width, cf.width_q, cf.width_h,
            cf.T_in, cf.T_out,
            cf.n_layers, cf.n_layers_q, cf.n_layers_h
        ).to(DEVICE)

    else:
        raise ValueError(f"Unknown NETWORK: {NETWORK}")

    num_params = count_params(model)
    print(f"Number of parameters: {num_params:,}")

    # -------------------------------------------------------------------------
    # 5) Setup optimizer, scheduler, and loss
    # -------------------------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=cf.learning_rate, weight_decay=cf.weight_decay)
    iterations = cf.epochs * (cf.nTrain // cf.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    loss_fn = LpLoss(size_average=False)

    # -------------------------------------------------------------------------
    # 6) Train the model
    # -------------------------------------------------------------------------
    import time
    start_time = time.time()

    if cf.training:
        if NETWORK == 'FNO2d':
            model, train_l2_log, test_l2_log = (
                train_fno_time(model, loss_fn, cf.epochs, cf.batch_size,
                               train_loader, test_loader,
                               optimizer, scheduler, cf.normalized,
                               normalizers, DEVICE)
            )
            # train_mse_log = []
        else:
            model, train_mse_log, train_l2_log, test_l2_log = train_fno(
                model, loss_fn, cf.epochs, cf.batch_size,
                train_loader, test_loader,
                optimizer, scheduler, cf.normalized,
                normalizers, DEVICE
            )
    else:
        raise RuntimeError("cf.training must be True for hyperparam sweep!")

    elapsed = time.time() - start_time
    print(f"  Training time: {elapsed:.1f} sec")

    # -------------------------------------------------------------------------
    # 7) Evaluate on test set
    # -------------------------------------------------------------------------
    evaluator = ModelEvaluator(
        model, dataset, test_dataset,
        cf.s, cf.T_in, cf.T_out, DEVICE,
        cf.normalized, normalizers,
        time_history=(NETWORK == 'FNO2d'),
        use_sliding_window=True
    )
    results = evaluator.evaluate(loss_fn=loss_fn)
    final_test_L2 = results["trajectory"]["average"]
    print(f"  Final test L2-error (average over trajectories): {final_test_L2:.6f}")

    # -------------------------------------------------------------------------
    # 8) Return results as dict
    # -------------------------------------------------------------------------
    return {
        'run_id': run_id,
        'hyperparams': param_dict,
        'results': results,
        'n_params': num_params,
        'final_test_L2': final_test_L2,
        'train_time_sec': elapsed,
        'model': model,
        'train_l2_log': train_l2_log,
        'test_l2_log': test_l2_log,
    }


# =============================================================================
# MAIN: ITERATE OVER GRID AND LOG RESULTS
# =============================================================================
if __name__ == '__main__':
    # 1) Build list of all hyperparameter combinations
    keys = list(HYPERPARAM_GRID.keys())
    all_combinations = list(itertools.product(*(HYPERPARAM_GRID[k] for k in keys)))
    total_runs = len(all_combinations)
    print(f"Total hyperparameter combinations: {total_runs}\n")

    # 2) Prepare CSV file to record results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"hyperparam_sweep_{PROBLEM}_{NETWORK}_Tout{T_OUT}_{PARAM}_{timestamp}.csv"
    csv_path = os.path.join(OUTPUT_DIR, output_file)
    csv_columns = [
        'run_id',
        *keys,
        'n_params',
        'final_test_L2',
        'train_time_sec',
    ]
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()

    # 3) Loop over all combinations
    for run_id, combo in enumerate(all_combinations, start=1):
        # Build a dict of {param_name: value}
        param_dict = {keys[i]: combo[i] for i in range(len(keys))}
        result = run_single_experiment(param_dict, run_id)
        # try:
        #     result = run_single_experiment(param_dict, run_id)
        # except Exception as e:
        #     print(f"Run {run_id} FAILED with error:\n{e}\n")
        #     continue

        # Flatten the hyperparams into the CSV row
        row = {
            'run_id': result['run_id'],
            **{k: result['hyperparams'][k] for k in keys},
            'n_params': result['n_params'],
            'final_test_L2': result['final_test_L2'],
            'train_time_sec': result['train_time_sec'],
        }

        # Append to CSV
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writerow(row)

        print(f"Run {run_id}/{total_runs} logged.\n")

    print(f"Hyperparameter sweep completed. Results saved to:\n  {csv_path}")
