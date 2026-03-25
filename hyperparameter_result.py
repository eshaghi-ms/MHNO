import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime


# ============================================================================
# SETTINGS — MUST MATCH YOUR SWEEP CONFIG
# ============================================================================
def as_datetime(path: Path) -> datetime:
    m = stamp.search(path.name)
    return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S") if m else datetime.min


PROBLEM = 'AC2D'
NETWORK = 'FNO2d'
PARAM = 'lr'
T_OUT = 10
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

CSV_FOLDER = Path(os.path.join(REPO_ROOT, PROBLEM, 'hyperparam_sweep_results'))
candidates = CSV_FOLDER.glob(f"hyperparam_sweep_{PROBLEM}_{NETWORK}_Tout{T_OUT}_{PARAM}_*.csv")

stamp = re.compile(fr"{PARAM}_(\d{{8}}_\d{{6}})\.csv$")

# Pick the newest by the timestamp embedded in the name
CSV_PATH = max(candidates, key=as_datetime)
# ============================================================================
# LOAD AND DISPLAY RESULTS
# ============================================================================
if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"CSV file not found at:\n  {CSV_PATH}")

# Read CSV
df = pd.read_csv(CSV_PATH)

# Sort by final test error (ascending)
df_sorted = df.sort_values(by='final_test_L2', ascending=True)

# Show top 10 runs
print("\nTop 10 configurations by lowest test L2-error:\n")
print(df_sorted.head(10).to_string(index=False))

# ============================================================================
# PLOT SETTINGS
# ============================================================================
sns.set(style="whitegrid")
plt.figure(figsize=(12, 6))

# ============================================================================
# PLOT 1: Final Test L2 Error vs Individual Hyperparameters
# ============================================================================
param_list = ['nTrjTrain', 'batch_size', 'learning_rate', 'weight_decay', 'epochs',
              'modes', 'width', 'width_q', 'width_h', 'n_layers', 'n_layers_q', 'n_layers_h', 'T_out']

for param in param_list:
    if param not in df.columns:
        continue
    plt.figure(figsize=(8, 5))
    if df[param].nunique() < 15:
        sns.boxplot(x=param, y='final_test_L2', data=df)
    else:
        sns.scatterplot(x=param, y='final_test_L2', data=df)
    plt.title(f'Final Test L2 Error vs {param}')
    plt.xlabel(param)
    plt.ylabel('Test L2 Error')
    plt.tight_layout()
    plt.grid(True)
    plt.show()

# ============================================================================
# PLOT 2: Correlation Heatmap (Optional)
# ============================================================================
# plt.figure(figsize=(10, 8))
# corr = df[param_list + ['final_test_L2']].corr()
# sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", center=0)
# plt.title("Correlation Between Hyperparameters and Test Error")
# plt.tight_layout()
# plt.show()
