import re
import os
import json
import torch
import numpy as np
import pandas as pd
from itertools import cycle
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, FormatStrFormatter
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter1d

problem = 'PFC2D'
# Directory where all model results are saved
RESULTS_DIR = os.path.join(problem, 'results')
PLOT_DIR = os.path.join(problem, 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# Find all the models we evaluated
model_basenames = sorted([
    fname.replace('_trajectory_stats.json', '')
    for fname in os.listdir(RESULTS_DIR)
    if fname.endswith('_trajectory_stats.json')
])


# Helper: load trajectory‐average L2 from the JSON
def load_trajectory_avg(basename):
    path = os.path.join(RESULTS_DIR, f'{basename}_trajectory_stats.json')
    with open(path, 'r') as f:
        stats = json.load(f)['average']
    return stats


# Helper to load the full list of per-trajectory L2 errors
def load_trajectory_stats(basename):
    path = os.path.join(RESULTS_DIR, f'{basename}_trajectory_stats.json')
    with open(path, 'r') as f:
        stats = json.load(f)
    return stats


# Helper: load per‐window stats into a DataFrame
# We expect columns: ['window_idx','count','average','std_dev','min','max','q1','median','q3']
def load_per_window_stats(basename):
    path = os.path.join(RESULTS_DIR, f'{basename}_per_window_stats.csv')
    return pd.read_csv(path)


# Fixed model labels and colors
fixed_labels = [
    "FNO2d-App. I", "FNO2d-App. II", "FNO2d-App. III",
    "FNO3d-App. I", "FNO3d-App. II", "FNO3d-App. III",
    "MHNO-App. I", "MHNO-App. II", "MHNO-App. III"
]

# Define colors for different approaches
colors = ['#FF00FF', '#00FFFF', '#FFFF00']
colors = colors * 3
# Build a dictionary of {model_basename: average_trajectory_L2}
traj_avgs = {base: load_trajectory_avg(base) for base in model_basenames}

# Collect per-window average L2 errors
model_data_per_window = []
for base in model_basenames:
    print(base)
    df = load_per_window_stats(base)
    model_data_per_window.append(df['average'].values)

# Sanity check to ensure label count matches data count
assert len(model_data_per_window) == len(fixed_labels), "Mismatch between model data and fixed labels."
######################################################
##########           Violin Plot           ###########
######################################################
plt.rcParams['font.family'] = "DejaVu Serif"
PLOT_NAME = os.path.join(PLOT_DIR, 'violin_plot_per_window.png')

# Prepare figure
plt.figure(figsize=(9, 5.2))

# Create horizontal violin plot with means, medians, and extrema
parts = plt.violinplot(
    dataset=model_data_per_window,
    vert=False,
    showmeans=True,
    showmedians=True,
    showextrema=False,
    widths=0.9,
)

# overlay quartile lines
for i, d in enumerate(model_data_per_window):
    q1, med, q3 = np.percentile(d, [25, 50, 75])
    plt.hlines(y=i + 1, xmin=q1, xmax=q3, color='black', linewidth=2)

# Apply colors to violins
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_edgecolor('black')
    pc.set_alpha(1)

# Set mean, median, and extrema colors to black
if 'cmeans' in parts:
    parts['cmeans'].set_color('black')
if 'cmedians' in parts:
    parts['cmedians'].set_color('black')
if 'cmins' in parts:
    parts['cmins'].set_color('black')
if 'cmaxes' in parts:
    parts['cmaxes'].set_color('black')

x_lim_min = 0.004
x_lim_max = 0.2
# Log scale for x-axis
plt.xscale('log')
plt.xlim(x_lim_min, x_lim_max)

# Remove y-tick labels
plt.yticks(ticks=range(1, 10), labels=[''] * 9)

ax = plt.gca()
ax.xaxis.set_major_locator(LogLocator(base=10))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2g'))

# Separator lines
plt.axhline(y=3.5, color='black', linestyle='--', linewidth=1)
plt.axhline(y=6.5, color='black', linestyle='--', linewidth=1)

# Section titles
plt.text(x_lim_min * 1.05, 3.05, 'FNO2d', va='center', fontsize=12, fontweight='bold')
plt.text(x_lim_min * 1.05, 6.1, 'FNO3d', va='center', fontsize=12, fontweight='bold')
plt.text(x_lim_min * 1.05, 9.5, 'MHNO', va='center', fontsize=12, fontweight='bold')

# Axis label and title
plt.xlabel('Average L2 Error (log scale)', fontsize=11)
plt.title('Model Comparison (Distribution of average L2 Error over time steps)')
# Note that the averaging is across different samples
# and the violin plot shows the distribution of these average values over different time steps
# Legend
legend_patches = [
    mpatches.Patch(color=colors[2], label='App. III'),
    mpatches.Patch(color=colors[1], label='App. II'),
    mpatches.Patch(color=colors[0], label='App. I'),
]
plt.legend(handles=legend_patches, loc='upper right', title='Approaches')

plt.tight_layout()
plt.savefig(PLOT_NAME, dpi=600, bbox_inches='tight')
# plt.show()
plt.close()
######################################################
##########           Violin Plot           ###########
##########         Full Trajectory         ###########
######################################################
# Plot name
plt.rcParams['font.family'] = "DejaVu Serif"
PLOT_NAME = os.path.join(PLOT_DIR, 'violin_plot_trajectory.png')

# Collect per-window average L2 errors
model_data_trajectory = []
for base in model_basenames:
    df = load_trajectory_stats(base)
    model_data_trajectory.append(df)


# Truncated normal sampling
def truncated_normal_samples(mean, scale, lower, upper, size):
    samples = []
    while len(samples) < size:
        batch = np.random.normal(loc=mean, scale=scale, size=size)
        valid = batch[(batch >= lower) & (batch <= upper)]
        samples.extend(valid.tolist())
    return np.array(samples[:size])


# Build dataset
data_for_violin = []
for entry in model_data_trajectory:
    iqr = entry['q3'] - entry['q1']
    scale = iqr / 1.35
    try:
        samples = truncated_normal_samples(entry['median'], scale, entry['min']['value'], entry['max']['value'], entry['count'])
    except:
        samples = truncated_normal_samples(entry['median'], scale, entry['min'], entry['max'], entry['count'])

    data_for_violin.append(samples)

# Prepare figure
plt.figure(figsize=(9, 5.2))

# Horizontal violin plot
parts = plt.violinplot(
    dataset=data_for_violin,
    vert=False,
    showmeans=True,
    showmedians=True,
    showextrema=False,
    widths=0.9,
)

# Overlay quartile lines
for i, d in enumerate(data_for_violin):
    q1, med, q3 = np.percentile(d, [25, 50, 75])
    plt.hlines(y=i + 1, xmin=q1, xmax=q3, color='black', linewidth=2)

# Apply colors
for i, pc in enumerate(parts['bodies']):
    pc.set_facecolor(colors[i])
    pc.set_edgecolor('black')
    pc.set_alpha(1)

# Set means/medians colors
if 'cmeans' in parts: parts['cmeans'].set_color('black')
if 'cmedians' in parts: parts['cmedians'].set_color('black')

# Axes scale and limits
x_lim_min = 0.004
x_lim_max = 0.2
plt.xscale('log')
plt.xlim(x_lim_min, x_lim_max)

# Remove y-tick labels
plt.yticks(ticks=range(1, 10), labels=[''] * 9)

ax = plt.gca()
ax.xaxis.set_major_locator(LogLocator(base=10))
ax.xaxis.set_major_formatter(FormatStrFormatter('%.2g'))

# Separator lines and section titles
plt.axhline(y=3.5, color='black', linestyle='--', linewidth=1)
plt.axhline(y=6.5, color='black', linestyle='--', linewidth=1)

plt.text(x_lim_min * 1.05, 3.05, 'FNO2d', va='center', fontsize=12, fontweight='bold')
plt.text(x_lim_min * 1.05, 6.1, 'FNO3d', va='center', fontsize=12, fontweight='bold')
plt.text(x_lim_min * 1.05, 9.5, 'MHNO', va='center', fontsize=12, fontweight='bold')

# Labels, title, legend
plt.xlabel('L2 Error (log scale)', fontsize=11)
plt.title('Model Comparison (Distribution of full trajectory L2 Error over different samples)')
# Note that there is no averaging here
# There is no time steps here (each values shows l2 norm for full trajectory)
# and the violin plot shows the distribution of full trajectory l2-norm values over different samples

legend_patches = [
    mpatches.Patch(color=colors[2], label='App. III'),
    mpatches.Patch(color=colors[1], label='App. II'),
    mpatches.Patch(color=colors[0], label='App. I'),
]
plt.legend(handles=legend_patches, loc='upper right', title='Approaches')

plt.tight_layout()
plt.savefig(PLOT_NAME, dpi=600, bbox_inches='tight')
# plt.show()
plt.close()

######################################################
##########            Line Plot            ###########
##########           Time  Steps           ###########
######################################################
plt.rcParams['font.family'] = "DejaVu Serif"
PLOT_NAME = os.path.join(PLOT_DIR, 'line_plot_l2_timestep.png')
fig_size = (6, 4)
plt.figure(figsize=fig_size)
colors = ['#FF0000', '#0000FF', '#000000']
y_lim_min = 0.004
y_lim_max = 0.5

# Define your color and line style mappings
method_colors = {
    "FNO2d": colors[0],
    "FNO3d": colors[1],
    "MHNO": colors[2]
}

approach_linestyles = {
    "App. I": ":",
    "App. II": "--",
    "App. III": "-"
}

x_new = 0
for i, base in enumerate(fixed_labels):
    df = load_per_window_stats(model_basenames[i])
    line_style = []
    color = []
    # Extract method and approach from base name
    for approach in approach_linestyles:
        if base.endswith(approach):
            line_style = approach_linestyles[approach]
            # break
    for method in method_colors:
        if base.startswith(method):
            color = method_colors[method]
            # break

    x = df['window_idx'].values
    y = df['average'].values

    # Interpolate for smooth curve
    if len(x) > 3:  # Need at least 4 points for spline
        x_new = np.linspace(x.min(), x.max(), 1000)
        spline = make_interp_spline(x, y, k=3)
        y_smooth = spline(x_new)
        plt.plot(x_new, y_smooth, label=base, color=color, linestyle=line_style, linewidth=3.0)
    else:
        plt.plot(x, y, label=base, color=color, linestyle=line_style, linewidth=3.0)

plt.xticks(np.arange(0, max(x_new) + 1, 10))
plt.xlim(0, 90)
plt.ylim(y_lim_min, y_lim_max)
plt.yscale('log')
plt.xlabel('Time Step', fontsize=11)
plt.ylabel('Average L2 Error', fontsize=11)
plt.title('Per-Time Step Error Comparison')

# Custom legend handles
approach_legend = [Line2D([0], [0], color='black', linestyle=ls, linewidth=2.0, label=method)
                  for method, ls in approach_linestyles.items()]
method_legend = [Line2D([0], [0], color=color, linestyle='-', linewidth=2.0, label=approach)
                for approach, color in method_colors.items()]

# Place legends side by side (bottom-right)
# first_legend = plt.legend(handles=method_legend, title="Method",
#                           loc='lower right', bbox_to_anchor=(1.00, 0.0))
# plt.gca().add_artist(first_legend)

# plt.legend(handles=approach_legend, title="Approach",
#            loc='lower right', bbox_to_anchor=(0.79, 0.0))
plt.legend(handles=approach_legend, title="Approach",
           loc='lower right', bbox_to_anchor=(1.00, -0.01))

# plt.grid(True)
plt.tick_params(axis='both', which='major', direction='in')
plt.minorticks_off()
plt.tight_layout()
plt.savefig(PLOT_NAME, dpi=600, bbox_inches='tight')
# plt.show()
plt.close()

"""
# 4) Save combined per-window stats into a single DataFrame, then CSV
#    (same as before, but now we assume CSV might already contain q1, median, q3)
first_df = load_per_window_stats(model_basenames[0])
combined = pd.DataFrame({'window_idx': first_df['window_idx']})

for base in model_basenames:
    df = load_per_window_stats(base)
    combined[f'{base}_avg']    = df['average']
    combined[f'{base}_std']    = df['std_dev']
    combined[f'{base}_min']    = df['min']
    combined[f'{base}_max']    = df['max']
    # If quartile columns exist, bring them in:
    if 'q1' in df.columns:
        combined[f'{base}_q1'] = df['q1']
    if 'median' in df.columns:
        combined[f'{base}_median'] = df['median']
    if 'q3' in df.columns:
        combined[f'{base}_q3'] = df['q3']

combined_csv = os.path.join(RESULTS_DIR, 'combined_per_window_stats.csv')
combined.to_csv(combined_csv, index=False)
print(f"Saved combined per-window stats to {combined_csv}")

"""
# 5) Print a text summary of trajectory averages
print("Trajectory-Level Summary:")
for base, avg in traj_avgs.items():
    print(f"{base}: average L2 = {avg:.6f}")

######################################################
##########       L2 error comparison       ###########
##########           Time  Steps           ###########
######################################################
plt.rcParams['font.family'] = "DejaVu Serif"
PLOT_NAME = os.path.join(PLOT_DIR, 'L2_error_comparison_timestep.png')

n_models = len(model_basenames)
rows, cols = 3, 3
fig, axes = plt.subplots(rows, cols, figsize=(12, 9), sharex=True, sharey=True)

colors = ['#F8766D',  # Q1 – coral
          '#0000FF',  # Median – pure blue
          '#00FF00',  # Q3 – neon green
          '#000000',  # Mean – solid black
          '#C77CFF']  # Std fill

axes = axes.flatten()
names = ['FNO2d - Appr. I', 'FNO2d - Appr. II', 'FNO2d - Appr. III',
         'FNO3d - Appr. I', 'FNO3d - Appr. II', 'FNO3d - Appr. III',
         'MHNO - Appr. I', 'MHNO - Appr. II', 'MHNO - Appr. III',]

for idx, (ax, base) in enumerate(zip(axes, model_basenames)):
    df = load_per_window_stats(base)
    idxs = df['window_idx']
    means = df['average']
    stds = df['std_dev']
    mins = df['min']
    maxs = df['max']

    # Optional: Q1, median, Q3
    q1 = df['q1']
    median = df['median']
    q3 = df['q3']
    ax.plot(idxs, q1, linestyle='--', color=colors[0], label='Q1', linewidth=2)
    ax.plot(idxs, median, linestyle='-', color=colors[1], label='Median', linewidth=2)
    ax.plot(idxs, q3, linestyle='--', color=colors[2], label='Q3', linewidth=2)

    # Plot mean
    ax.plot(idxs, means, color=colors[3], linewidth=3, label='Mean')

    # Shade ±1 std‐dev
    # lower = np.maximum(means - stds, mins)
    lower = means - stds
    upper = means + stds
    ax.fill_between(idxs, lower, upper, color=colors[4], alpha=0.25, label='±1 Std Dev')

    # ax.plot(idxs, mins, linestyle='--', color='C1', label='Min L2')
    # ax.plot(idxs, maxs, linestyle='--', color='C2', label='Max L2')

    # Short name inside subplot
    # parts = base.split('_')
    # short_name = f"{parts[0]}_{parts[3]}"
    # ax.text(0.02, 0.95, names[idx], transform=ax.transAxes,
    #         fontsize=11, fontweight='bold', va='top', ha='left')

    if idx == 2:
        ax.legend(loc='lower left', fontsize=11)

    # ax.set_ylim(0.0, 0.2)
    ax.set_ylim(0.005, 0.9)
    ax.set_xlim(0, 90)
    ax.set_yscale('log')
    ax.set_xticks(range(0, 100, 10))
    ax.set_xticklabels(range(0, 100, 10))

    # Set custom y-ticks and labels
    # ax.set_yticks(np.arange(0, 0.22, 0.02))
    # ax.set_yticklabels(np.arange(0, 0.22, 0.02))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
    # ax.grid(True)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.tick_params(axis='both', which='major', direction='in')

    if idx % cols == 0:
        ax.set_ylabel('L2 Error')
    if idx // cols == rows - 1:
        ax.set_xlabel('Time Step')

# Remove extra subplots if any
for j in range(n_models, len(axes)):
    fig.delaxes(axes[j])

fig.suptitle('Per-Window Error Stats: Mean ± Std Dev, Quartiles (One Subplot per Model)', y=0.94, fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig(PLOT_NAME, dpi=600, bbox_inches='tight')
# plt.show()
plt.close()
######################################################
# 7) Single subplot that overlays all models’ mean ± std bands
plt.figure(figsize=(10, 6))

for base in model_basenames:
    df = load_per_window_stats(base)
    idxs = df['window_idx']
    means = df['average']
    stds = df['std_dev']

    # Short name for legend
    parts = base.split('_')
    short_name = f"{parts[0]}_{parts[3]}"

    # Plot mean line
    plt.plot(idxs, means, label=short_name)

    # Shade ±1 std‐dev
    lower = means - stds
    upper = means + stds
    plt.fill_between(idxs, lower, upper, alpha=0.2)

plt.xlabel('Window Index')
plt.ylabel('L2 Error')
plt.title('Per-Window Error: All Models Overlayed (Mean ± Std Dev)')
plt.ylim(0.0, 0.50)
plt.xlim(80, 90)
plt.legend(loc='upper right', fontsize='small')
# plt.grid(True)
plt.minorticks_off()
plt.tick_params(axis='both', which='major', direction='in')
plt.tight_layout()
# plt.show()
plt.close()

####################################################################
####################################################################
####################################################################
####################################################################
# ----------------------------------------------------------------------------
colors = ['#0000FF', '#FF0000', '#FFFF00']


def plot_box_approach(selected_approach, windows, fixed):
    filename = f"{selected_approach}-{windows[0]}to{windows[-1]}.png"
    PLOT_NAME = os.path.join(PLOT_DIR, filename)
    # 1) keep only basenames having that approach
    filtered_basenames = [
        b for b in model_basenames
        if b.split('_')[3] == selected_approach
    ]

    # 2) now use filtered_basenames everywhere instead of model_basenames
    short_names = []
    for base in filtered_basenames:
        parts = base.split('_')
        short_names.append(f"{parts[0]}_{parts[3]}")

    n_models = len(filtered_basenames)
    n_windows = len(windows)
    group_gap = 1  # gap between window groups
    stride = n_models + group_gap

    # Precompute every model’s stats & positions
    all_stats = []
    all_positions = []
    for i, win in enumerate(windows):
        for j, base in enumerate(filtered_basenames):
            df = load_per_window_stats(base)
            row = df.loc[df['window_idx'] == win].iloc[0]
            stat = {
                'med': row['median'],
                'q1': row['q1'],
                'q3': row['q3'],
                'whislo': row['min'],
                # 'whislo': row['median'] - row['std_dev'],
                'whishi': row['max'],
                # 'whishi': row['median'] + row['std_dev'],
                'label': short_names[j]
            }
            all_stats.append(stat)
            all_positions.append(i * stride + j)

    # 1) Set up figure
    plt.rcParams['font.family'] = "DejaVu Serif"
    fig, ax = plt.subplots(figsize=(10, 4.5))  # double-column width

    # 2) Prepare a color map
    # colors = plt.get_cmap('tab10').colors

    # 3) Draw each model’s boxes in its own call
    for j in range(n_models):
        # slice out this model’s stats & positions:
        stats_j = all_stats[j::n_models]
        positions_j = all_positions[j::n_models]

        bp = ax.bxp(
            stats_j,
            positions=positions_j,
            widths=0.8,
            showfliers=False,
            patch_artist=True
        )

        # color just this batch of boxes:
        for box in bp['boxes']:
            box.set_facecolor(colors[j])
            box.set_edgecolor('black')
            box.set_alpha(0.7)

        for median in bp['medians']:
            median.set_color('black')
            median.set_linewidth(1.5)
        # you can also color whiskers/caps in this loop if you like

    # 4) Tidy up axes & legend
    group_centers = [i * stride + (n_models - 1) / 2 for i in range(len(windows))]
    ax.set_xticks(group_centers)
    ax.set_xticklabels(windows, fontsize=9)
    ax.set_xlim(-group_gap, all_positions[-1] + group_gap)
    ax.set_yscale('log')
    if fixed is not False:
        ax.set_ylim(fixed[0], fixed[-1])

    ax.set_xlabel('Time Step', fontsize=11)
    ax.set_ylabel('L2-Error', fontsize=11)
    # ax.set_title('Per-Window L₂ Error Distributions (Box Plots)', fontsize=12, fontweight='bold')

    legend_handles = [
        plt.Line2D([0], [0], color=colors[j], lw=10, alpha=0.7, label=short_names[j])
        for j in range(n_models)
    ]
    # ax.legend(handles=legend_handles,
    #           loc='center left',
    #           bbox_to_anchor=(1.02, 0.5),
    #           fontsize=9,
    #           title='Model')

    plt.tight_layout()
    plt.tick_params(axis='both', which='major', direction='in', labelsize=16)
    plt.minorticks_off()
    plt.savefig(PLOT_NAME, dpi=600, bbox_inches='tight')
    plt.show()


# 0) figure out which “approach” string is the first one
windows = list(range(1, 11))
selected_approach = model_basenames[0].split('_')[3]  # "approach1"
plot_box_approach(selected_approach, windows, [0.0001, 1])

selected_approach = model_basenames[1].split('_')[3]  # "approach2"
plot_box_approach(selected_approach, windows, [0.0007, 0.8])

selected_approach = model_basenames[2].split('_')[3]  # "approach3"
plot_box_approach(selected_approach, windows, [0.003, 1])

windows = list(range(81, 91))
selected_approach = model_basenames[0].split('_')[3]  # "approach1"
plot_box_approach(selected_approach, windows, [0.0001, 1])

selected_approach = model_basenames[1].split('_')[3]  # "approach2"
plot_box_approach(selected_approach, windows, [0.0007, 0.8])

selected_approach = model_basenames[2].split('_')[3]  # "approach3"
plot_box_approach(selected_approach, windows, [0.003, 1])


###################################################################
###################################################################
###################################################################
###################################################################
###################################################################
def extract_model_name(filename):
    """Extract model name (e.g., FNO2d) from a filename like 'losses_FNO2d_...'."""
    parts = filename.split('_')
    return parts[1] if len(parts) > 1 else filename


def load_and_plot_loss(log_files, approach_tag, y_limit):
    """
    Plot losses from multiple files with separate legends:
      - Color for each method (FNO2d, etc.)
      - Linestyle for Train/Test
    """
    filename = f"{approach_tag}-LossComparison.png"
    PLOT_NAME = os.path.join(PLOT_DIR, filename)
    # colors = cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    colors = cycle(['#FF0000', '#0000FF', '#000000'])

    method_to_color = {}
    custom_lines = []

    plt.figure(figsize=fig_size)
    for path in log_files:
        logs = torch.load(path)
        train_l2 = logs.get('train_l2_log', [])
        test_l2 = logs.get('test_l2_log', [])
        filename = os.path.basename(path)
        method = extract_model_name(filename)

        if method not in method_to_color:
            color = next(colors)
            method_to_color[method] = color
            custom_lines.append(Line2D([0], [0], color=color, lw=2, label=method))

        color = method_to_color[method]
        plt.plot(train_l2, color=color, linestyle='-', label=f'{method} Train')
        plt.plot(test_l2, color=color, linestyle='--', label=f'{method} Test')

    # Create dummy lines for Train/Test legend
    style_lines = [
        Line2D([0], [0], color='black', lw=2, linestyle='-', label='Train'),
        Line2D([0], [0], color='black', lw=2, linestyle='--', label='Test')
    ]

    plt.xlabel('Epoch')
    plt.ylabel('L2 Loss')
    plt.yscale('log')
    plt.ylim(y_limit)
    plt.title('Comparison of Loss Curves')
    plt.minorticks_off()
    plt.tick_params(axis='both', which='major', direction='in')
    # plt.grid(True)

    # Place two legends: one for method (colors), one for Train/Test (linestyles)
    # legend1 = plt.legend(handles=custom_lines, bbox_to_anchor=(0.77, 1), loc='upper left')
    # legend2 = plt.legend(handles=style_lines, bbox_to_anchor=(0.58, 1), loc='upper left')
    legend2 = plt.legend(handles=style_lines, bbox_to_anchor=(0.765, 1), loc='upper left')
    plt.gca().add_artist(legend2)

    plt.tight_layout()
    plt.savefig(PLOT_NAME, dpi=600, bbox_inches='tight')
    # plt.show()


def find_and_plot_by_approach(results_dir, approach_tag, y_limit):
    """
    Scan results_dir for any files starting with 'losses' that also contain
    the approach_tag substring, load them, and plot.
    """
    if not os.path.isdir(results_dir):
        raise ValueError(f"Directory not found: {results_dir}")

    matching = [
        os.path.join(results_dir, fn)
        for fn in os.listdir(results_dir)
        if fn.startswith("losses") and approach_tag in fn
    ]

    if not matching:
        print(f"No loss files matching approach '{approach_tag}' in {results_dir}")
        return

    print(f"Found {len(matching)} files for approach '{approach_tag}':")
    for path in matching:
        print("  ", os.path.basename(path))

    load_and_plot_loss(matching, approach_tag, y_limit)


# === USAGE ===
results_dir = os.path.join(problem, 'results')
approach = "T1to10"  # or "T1to90", etc.
find_and_plot_by_approach(results_dir, approach, (0.001, 0.7))
approach = "T1to45"  # or "T1to90", etc.
find_and_plot_by_approach(results_dir, approach, (0.002, 0.7))
approach = "T1to90"  # or "T1to90", etc.
find_and_plot_by_approach(results_dir, approach, (0.01, 1.1))
