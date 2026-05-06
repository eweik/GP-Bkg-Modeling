
#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import warnings

# Suppress standard SciPy warnings globally just in case
warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)

# Setup paths to import local modules
current_script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_script_dir)
if repo_root not in sys.path: sys.path.append(repo_root)
if os.getcwd() not in sys.path: sys.path.append(os.getcwd())

from src.config import ATLAS_BINS

def get_gp_fit(centers, density, density_err, min_len_scale_log=0.15):
    """Calculates the zero-mean GP fit."""
    mask = density > 0
    X_log = np.log(centers[mask]).reshape(-1, 1)
    y_target = np.log(density[mask])
    y_err_target = density_err[mask] / density[mask]

    kernel = C(1.0, (1e-3, 1e2)) * RBF(length_scale=0.3, length_scale_bounds=(min_len_scale_log, 5.0))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=y_err_target**2, n_restarts_optimizer=5, normalize_y=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X_log, y_target)

    X_full_log = np.log(centers).reshape(-1, 1)
    y_pred_target = gp.predict(X_full_log)

    return np.exp(y_pred_target)

def get_channel_data_gp(base_dir, trigger, channel, cms, mass_matrix, col_names, min_len):
    """Extracts data, fits the GP, and builds the discrete CDF and bounds."""
    fitfile = os.path.join(base_dir, "fits", f"fitme_p5_{trigger}_{channel}.json")
    if not os.path.exists(fitfile):
        return None

    with open(fitfile, "r") as j_nom:
        d_nom = json.load(j_nom)

    fmin_val, fmax_val = float(d_nom['fmin']), float(d_nom['fmax'])
    v_bins = ATLAS_BINS[(ATLAS_BINS >= fmin_val) & (ATLAS_BINS <= fmax_val)]
    c = (v_bins[:-1] + v_bins[1:]) / 2
    widths = np.diff(v_bins)

    idx = col_names.index(f"M{channel}")
    valid_masses = mass_matrix[mass_matrix[:, idx] > 0, idx] * cms

    # 1. Histogram raw data to train GP
    data_counts, _ = np.histogram(valid_masses, bins=v_bins)
    bkg_density = data_counts / widths
    bkg_err_density = np.sqrt(np.maximum(data_counts, 1.0)) / widths

    # 2. Evaluate GP Fit (The Null Hypothesis)
    gp_density = get_gp_fit(c, bkg_density, bkg_err_density, min_len)
    counts_gp = gp_density * widths

    # 3. Create Discrete CDF *FROM THE GP*
    cdf_gp = np.cumsum(counts_gp) / np.sum(counts_gp)

    # 4. Calculate Phase-Space Truncation Bounds in Uniform Space
    N_valid = len(valid_masses)
    if N_valid > 0:
        u_min = np.sum(valid_masses < fmin_val) / N_valid
        u_max = np.sum(valid_masses <= fmax_val) / N_valid
    else:
        u_min, u_max = 0.0, 1.0

    return cdf_gp, v_bins, c, (fmin_val, fmax_val), (u_min, u_max)

def map_uniform_to_mass(u_array, u_bounds, cdf, centers, apply_jitter=False):
    """Maps uniform [0,1] variables to discrete bins, with optional jitter."""
    u_min, u_max = u_bounds

    if apply_jitter:
        u_array = u_array + np.random.uniform(-0.0002, 0.0002, size=len(u_array))

    u_trunc = (u_array - u_min) / max(u_max - u_min, 1e-10)
    u_trunc = np.abs(u_trunc)
    u_trunc = np.where(u_trunc >= 1.0, 1.99999 - u_trunc, u_trunc)

    idx = np.searchsorted(cdf, u_trunc)
    idx = np.clip(idx, 0, len(centers) - 1)

    return centers[idx]

def safe_spearman(x, y):
    """Safely calculates Spearman rank correlation, returning 0.0 for zero-variance arrays."""
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return 0.0

    s, _ = stats.spearmanr(x, y)

    if np.isnan(s):
        return 0.0

    return s

def main():
    parser = argparse.ArgumentParser(description="Plot Full Spearman Matrix: Raw vs Copula (GP Background)")
    parser.add_argument('--trigger', type=str, required=True, help="Trigger name")
    parser.add_argument('--cms', type=float, default=13000., help="Center of mass energy")
    parser.add_argument('--min-len', type=float, default=0.15, help="GP Min Length Scale")
    args = parser.parse_args()

    base_dir = os.getcwd() if os.path.exists("data") else repo_root

    print("Loading data matrices...")
    mass_path = os.path.join(base_dir, "data", f"masses_{args.trigger}.npz")
    copula_path = os.path.join(base_dir, "data", f"copula_{args.trigger}.npz")

    if not os.path.exists(mass_path) or not os.path.exists(copula_path):
        print("Error: Could not find masses or copula npz files.")
        sys.exit(1)

    f_mass = np.load(mass_path)
    f_copula = np.load(copula_path)
    mass_matrix = f_mass['masses']
    copula_matrix = f_copula['copula']

    # FIX 1: Extract both sets of column names for mapping
    col_names_mass = list(f_mass['columns'])
    col_names_cop = list(f_copula['columns'])

    n_cols = len(col_names_mass)

    channel_info = {}
    print("Fitting GPs and calculating phase-space bounds for all channels...")
    for i, col in enumerate(col_names_mass):
        channel = col.replace("M", "")
        data = get_channel_data_gp(base_dir, args.trigger, channel, args.cms, mass_matrix, col_names_mass, args.min_len)

        if data is None:
            print(f"Warning: Missing fit limits for {channel}. Skipping in matrix.")
            continue

        cdf_gp, bins, centers, mass_bounds, u_bounds = data

        channel_info[i] = {
            'cdf': cdf_gp, 'centers': centers,
            'mass_bounds': mass_bounds, 'u_bounds': u_bounds
        }

    corr_raw = np.eye(n_cols)
    corr_copula = np.eye(n_cols)

    print("Calculating pairwise Spearman Rank correlations...")
    # FIX 2: Massively oversample to ensure survival in the deep tails
    N_toys = 10_000_000

    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            if i not in channel_info or j not in channel_info:
                continue

            info_i, info_j = channel_info[i], channel_info[j]
            col_i, col_j = col_names_mass[i], col_names_mass[j]

            # --- 1. RAW DATA (In-Window) ---
            m_i_all = mass_matrix[:, i] * args.cms
            m_j_all = mass_matrix[:, j] * args.cms

            valid_raw_mask = (m_i_all >= info_i['mass_bounds'][0]) & (m_i_all <= info_i['mass_bounds'][1]) & \
                             (m_j_all >= info_j['mass_bounds'][0]) & (m_j_all <= info_j['mass_bounds'][1])

            m_i_raw = m_i_all[valid_raw_mask]
            m_j_raw = m_j_all[valid_raw_mask]

            corr_raw[i, j] = corr_raw[j, i] = safe_spearman(m_i_raw, m_j_raw)

            # --- 2. COPULA TOYS (Mapped to Gaussian Process) ---

            # FIX 3: Map the indices safely!
            idx_cop_i = col_names_cop.index(col_i)
            idx_cop_j = col_names_cop.index(col_j)

            random_indices = np.random.choice(len(copula_matrix), size=N_toys, replace=True)
            u_i_sampled = copula_matrix[random_indices, idx_cop_i]
            u_j_sampled = copula_matrix[random_indices, idx_cop_j]

            u_min_i, u_max_i = info_i['u_bounds']
            u_min_j, u_max_j = info_j['u_bounds']

            survivor_mask = (u_i_sampled >= u_min_i) & (u_i_sampled <= u_max_i) & \
                            (u_j_sampled >= u_min_j) & (u_j_sampled <= u_max_j)

            u_i_survivors = u_i_sampled[survivor_mask]
            u_j_survivors = u_j_sampled[survivor_mask]

            # Map valid survivors to physical mass bins using GP CDF
            m_i_toy = map_uniform_to_mass(u_i_survivors, info_i['u_bounds'], info_i['cdf'], info_i['centers'], apply_jitter=True)
            m_j_toy = map_uniform_to_mass(u_j_survivors, info_j['u_bounds'], info_j['cdf'], info_j['centers'], apply_jitter=True)

            corr_copula[i, j] = corr_copula[j, i] = safe_spearman(m_i_toy, m_j_toy)

    # --- 3. PLOTTING ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    vmin, vmax = -0.1, 1.0

    sns.heatmap(corr_raw, ax=axes[0], cmap=cmap, vmin=vmin, vmax=vmax,
                xticklabels=col_names_mass, yticklabels=col_names_mass,
                annot=True, fmt=".2f", square=True, cbar_kws={"shrink": .8})
    axes[0].set_title(f"Raw Data Spearman $\\rho$ (In-Window)\n{args.trigger.upper()}", fontsize=14)

    sns.heatmap(corr_copula, ax=axes[1], cmap=cmap, vmin=vmin, vmax=vmax,
                xticklabels=col_names_mass, yticklabels=col_names_mass,
                annot=True, fmt=".2f", square=True, cbar_kws={"shrink": .8})
    axes[1].set_title(f"Copula Toys Spearman $\\rho$ (Mapped to GP Null)\n{args.trigger.upper()}", fontsize=14)

    plt.suptitle("Comparison of Global Rank Correlation (Spearman) via GP", fontsize=18, fontweight='bold', y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.join(base_dir, "plots"), exist_ok=True)
    out_path = os.path.join(base_dir, "plots", f"full_spearman_matrix_comparison_gp_{args.trigger}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"\nSuccessfully saved GP Spearman correlation matrix plot to: {out_path}")

if __name__ == "__main__":
    main()
