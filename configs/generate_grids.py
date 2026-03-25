#!/usr/bin/env python3
"""
generate_grids.py  –  create hyper‑parameter sweep JSON files

Changes (2025‑07‑06)
--------------------
* `nTrjTrain` is now set per‑(problem, T_out) as requested:
      AC2D:  10→5,   45→8,    90→180
      all others: 10→8, 45→12, 90→420
* Removed the fragile `relative_to(Path.cwd())` print.
"""

from copy import deepcopy
import importlib
import json
from pathlib import Path

# ----------------------------------------------------------------------
# sweep definitions ----------------------------------------------------
# ----------------------------------------------------------------------
PROBLEMS = ["AC2D", "CH2D", "SH2D", "PFC2D", "MBE2D"]
NETS = ["FNO2d", "FNO3d", "TNO2d"]
T_OUTS = ["90", "45", "10"]

# Param ranges to sweep
SWEEPS = {
    "lr": {"learning_rate": [0.005, 0.001, 0.0005, 0.0001, 0.00005]},
    "modes": {"modes": [6, 8, 10, 12, 14, 16, 18]},
    "width": {"width": [12, 16, 20, 24, 32, 64]},
    "width_q": {"width_q": [12, 16, 20, 24, 32, 64, 128]},
    "width_h": {"width_h": [0, 16, 32, 64]},
    "n_layers": {"n_layers": [2, 4, 6, 8]},
    "n_layers_q": {"n_layers_q": [2, 4, 6, 8]},
    "n_layers_h": {"n_layers_h": [0, 2, 4]},
}

# New per‑case training‑set sizes
NTRJ_TRAIN = {
    "AC2D": {"10": 5, "45": 8, "90": 180},
    # Default for all other problems
    "_": {"10": 8, "45": 12, "90": 420},
}


# ----------------------------------------------------------------------
# helpers --------------------------------------------------------------
# ----------------------------------------------------------------------
def load_cfg(problem: str, net: str) -> dict:
    """Import config_<PROBLEM>_<NET>.py and expose its public attributes."""
    mod = importlib.import_module(f"config_{problem}_{net}")
    return {k: getattr(mod, k) for k in dir(mod) if not k.startswith("_")}


def base_grid(cfg: dict, t_out: int) -> dict:
    """Constant part of every grid (wrap scalars in single‑element lists)."""
    KEYS = [
        "nTrjTrain", "nTrjTest", "batch_size",
        "learning_rate", "weight_decay", "epochs",
        "modes", "width", "width_q", "width_h",
        "n_layers", "n_layers_q", "n_layers_h"
    ]
    g = {k: [cfg[k]] for k in KEYS if k in cfg}
    g["T_out"] = [t_out]
    return g


def write_grid(path: Path, grid: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(grid, indent=2))
    print(f"wrote {path}")


# ----------------------------------------------------------------------
# main -----------------------------------------------------------------
# ----------------------------------------------------------------------
def main():
    for problem in PROBLEMS:
        for net in NETS:

            # Allowed parameters to sweep for this network
            if net == "TNO2d":
                params = ["modes", "width", "width_q", "n_layers", "n_layers_q"]
            else:
                params = ["modes", "width", "width_q", "width_h",
                          "n_layers", "n_layers_q", "n_layers_h"]

            cfg = load_cfg(problem, net)

            for t_out in T_OUTS:
                # Select proper nTrjTrain override
                nt_map = NTRJ_TRAIN.get(problem, NTRJ_TRAIN["_"])
                cfg_mod = cfg.copy()
                cfg_mod["nTrjTrain"] = nt_map[t_out]

                base = base_grid(cfg_mod, int(t_out))

                for p in ["lr", *params]:
                    sweep = SWEEPS[p]
                    grid = deepcopy(base)

                    if p == "lr":
                        grid.update(sweep)  # vary learning rate
                    else:
                        grid["learning_rate"] = [cfg["learning_rate"]]
                        grid.update(sweep)  # vary chosen param

                    fname = Path("grids") / f"grid_{problem}_{net}_Tout{t_out}_{'lr' if p == 'lr' else p}.json"
                    write_grid(fname, grid)


if __name__ == "__main__":
    main()
