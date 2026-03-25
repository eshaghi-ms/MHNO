#!/usr/bin/env python
"""Submit hyper‑parameter sweep jobs for multiple problem / network / t_out combinations."""

import subprocess
from itertools import product

PROBLEMS = ["AC2D", "CH2D", "SH2D", "PFC2D", "MBE2D"]
NETS     = ["FNO2d", "FNO3d", "TNO2d"]
T_OUTS   = ["90", "45", "10"]


LOG_FMT = "output_HP_{ds}_{arch}_{rid}_{p}.log"
ERR_FMT = "error_HP_{ds}_{arch}_{rid}_{p}.log"
PY      = "/home/zore8312/miniforge3/envs/torch/bin/python"
SCRIPT  = "/scratch/zore8312/PycharmProjects/MHFNO/hyperparameter_sweep.py"

for problem, net, t_out in product(PROBLEMS, NETS, T_OUTS):
    if NETS == "TNO2d":
        PARAMS = ["modes", "width", "width_q", "width_h", "n_layers", "n_layers_q", "n_layers_h"]
    else:
        PARAMS = ["modes", "width", "width_q", "n_layers", "n_layers_q"]

    for param in PARAMS:
        out = LOG_FMT.format(ds=problem, arch=net, rid=t_out, p=param)
        err = ERR_FMT.format(ds=problem, arch=net, rid=t_out, p=param)
        cmd = (
            f'bsub -gpu "num=1:mode=exclusive_process" -q BatchGPU '
            f'-o {out} -e {err} '
            f'{PY} {SCRIPT} --problem {problem} --network {net} --t_out {t_out} '
            f'--param {param} --gpu "cuda:0"'
        )
        subprocess.run(cmd, shell=True, check=True)
