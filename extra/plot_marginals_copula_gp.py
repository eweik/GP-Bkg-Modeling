#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import warnings

# Suppress standard SciPy warnings globally
warnings.filterwarnings("ignore")

# Setup paths to import local modules
current_script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_script_dir)
if repo_root not in sys.path: sys.path.append(repo_root)
if os.getcwd() not in sys.path: sys.path.append(os.getcwd())

from src.config import ATLAS_BINS
from src.models import FiveParam

def get_gp_fit(centers, density, density_err, min_len_scale_log=0.15):
    """Calculates the zero-mean GP fit to serve as the new baseline."""
    mask = density > 0
    X_log = np.log(centers[mask]).reshape(-1, 1)
    y_target = np.log(density[mask])
    y_err_target = density_err[mask] / density[mask]

    kernel = C(1.0, (1e-3, 1e2)) * RBF(length_scale=0.3, length_scale_bounds=(min_len_scale_log, 5.0))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=y_err_target**2, n_restarts_optimizer=5, normalize_y=False)
    gp.fit(X_log, y_target)
        
    X_full_log = np.log(centers).reshape(-1, 1)
    y_pred_target = gp.predict(X_full_log)
    
    return np.exp(y_pred_target)

def get_channel_data_gp(base_dir, trigger, channel, cms, mass_matrix, col_names, min_len):
    """Extracts data, fits the GP, and builds the discrete CDF based on the GP."""
    idx = col_names.index(f"M{channel}")
    masses = mass_matrix[:, idx]
    valid_masses = masses[masses > 0] * cms
    
    fitfile = os.path.join(base_dir, "fits", f"fitme_p5_{trigger}_{channel}.json")
    if not os.path.exists(fitfile):
        raise FileNotFoundError(f"Fit file not found: {fitfile}")

    with open(fitfile, "r") as j_nom:
        d_nom = json.load(j_nom)
        
    fmin_val, fmax_val = float(d_nom['fmin']), float(d_nom['fmax'])
    v_bins = ATLAS_BINS[(ATLAS_BINS >= fmin_val) & (ATLAS_BINS <= fmax_val)]
    c = (v_bins[:-1] + v_bins[1:]) / 2
    widths = np.diff(v_bins)
    
    # 1. Histogram raw data
    data_counts, _ = np.histogram(valid_masses, bins=v_bins)
    bkg_density = data_counts / widths
    bkg_err_density = np.sqrt(np.maximum(data_counts, 1.0)) / widths
    
    # 2. Evaluate Reference 5-Param Fit (for plotting comparison)
    counts_5p = FiveParam(cms, c, *[float(p) for p in d_nom['parameters']]) 
    counts_5p = np.maximum(counts_5p, 0)
    
    # 3. Evaluate GP Fit (The NEW Null Hypothesis)
    gp_density = get_gp_fit(c, bkg_density, bkg_err_density, min_len)
    counts_gp = gp_density * widths
    
    # 4. Create Discrete CDF *FROM THE GP*
    cdf_gp = np.cumsum(counts_gp) / np.sum(counts_gp)

    # 5. Calculate Phase-Space Truncation Bounds in Uniform Space
    N_valid = len(valid_masses)
    if N_valid > 0:
        u_min = np.sum(valid_masses < v_bins[0]) / N_valid
        u_max = np.sum(valid_masses <= v_bins[-1]) / N_valid
    else:
        u_min, u_max = 0.0, 1.0
        
    return data_counts, counts_5p, counts_gp, cdf_gp, v_bins, c, (u_min, u_max)

def generate_expected_gp_copula_marginal(copula_matrix, col_names, channel, u_bounds, cdf, centers, bins):
    """Generates the averaged toy marginal by mapping copula samples through the GP CDF."""
    idx = col_names.index(f"M{channel}")
    N_data = len(copula_matrix)
    
    # Simulate N_mult full pseudo-experiments to get a smooth average line
    N_mult = 1000
    N_draw_total = N_data * N_mult
    
    sampled_rows = copula_matrix[np.random.choice(N_data, size=N_draw_total, replace=True)]
    u_raw = sampled_rows[sampled_rows[:, idx] >= 0, idx]
    
    u_min, u_max = u_bounds
    mask_in_window = (u_raw >= u_min) & (u_raw <= u_max)
    u_in_window = u_raw[mask_in_window]
    
    if len(u_in_window) == 0:
        return np.zeros(len(bins) - 1)
        
    u_jittered = u_in_window + np.random.uniform(-0.0002, 0.0002, size=len(u_in_window))
    
    # Local mapping
    u_trunc = (u_jittered - u_min) / max(u_max - u_min, 1e-10)
    u_trunc = np.abs(u_trunc)
    u_trunc = np.where(u_trunc >= 1.0, 1.99999 - u_trunc, u_trunc)
    
    # Map to physical mass via the GP's discrete CDF
    idx_mapped = np.searchsorted(cdf, u_trunc)
    idx_mapped = np.clip(idx_mapped, 0, len(centers) - 1)
    m_toy = centers[idx_mapped]
    
    toy_counts_total, _ = np.histogram(m_toy, bins=bins)
    expected_toy_counts = toy_counts_total / N_mult
    
    return expected_toy_counts

def main():
    parser = argparse.ArgumentParser(description="Plot Marginal Agreement: Data vs GP vs Copula Toys")
    parser.add_argument('--trigger', type=str, required=True, help="Trigger name (e.g., t2)")
    parser.add_argument('--ch1', type=str, default='jj', help="First channel to plot")
    parser.add_argument('--ch2', type=str, default='jb', help="Second channel to plot")
    parser.add_argument('--cms', type=float, default=13000., help="Center of mass energy")
    parser.add_argument('--min-len', type=float, default=0.15, help="GP Min Length Scale")
    args = parser.parse_args()

    base_dir = os.getcwd() if os.path.exists("data") else repo_root

    print("Loading data matrices...")
    mass_path = os.path.join(base_dir, "data", f"masses_{args.trigger}.npz")
    copula_path = os.path.join(base_dir, "data", f"copula_{args.trigger}.npz")
    
    f_mass = np.load(mass_path)
    f_copula = np.load(copula_path)
    mass_matrix = f_mass['masses']
    copula_matrix = f_copula['copula']
    col_names = list(f_mass['columns'])

    print("Fitting GPs and calculating mapping logic...")
    d1, c5p_1, cGP_1, cdf_gp_1, bins1, centers1, bounds1 = get_channel_data_gp(base_dir, args.trigger, args.ch1, args.cms, mass_matrix, col_names, args.min_len)
    d2, c5p_2, cGP_2, cdf_gp_2, bins2, centers2, bounds2 = get_channel_data_gp(base_dir, args.trigger, args.ch2, args.cms, mass_matrix, col_names, args.min_len)

    print("Generating GP-mapped Copula toys...")
    expected_toys1 = generate_expected_gp_copula_marginal(copula_matrix, col_names, args.ch1, bounds1, cdf_gp_1, centers1, bins1)
    expected_toys2 = generate_expected_gp_copula_marginal(copula_matrix, col_names, args.ch2, bounds2, cdf_gp_2, centers2, bins2)

    print("Generating plot...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    def plot_marginal(ax, channel, bins, centers, data, counts_5p, counts_gp, toys):
        
        # 1. 5-Param Reference (Dashed Blue)
        ax.plot(centers, counts_5p, color='dodgerblue', linestyle='--', linewidth=2, alpha=0.7, label='Analytic 5-Param (Reference)', zorder=2)
        
        # 2. GP Baseline Expectation (Solid Green)
        ax.plot(centers, counts_gp, color='forestgreen', linewidth=2.5, alpha=0.9, label='Gaussian Process Mean (New Null)', zorder=3)
        
        # 3. GP-Copula Toys (Red Dots)
        ax.plot(centers, toys, color='red', marker='o', linestyle='none', markersize=6, alpha=0.9, markeredgecolor='darkred', label='GP-Copula Toys (Expected)', zorder=4)
        
        # 4. Raw Data (Black Points)
        err = np.sqrt(data)
        err[err == 0] = 1.0 
        ax.errorbar(centers, data, yerr=err, fmt='ko', markersize=4, capsize=3, label='Raw Data', zorder=10)

        ax.set_title(f"Marginal Agreement: {channel.upper()} Channel", fontsize=15)
        ax.set_xlabel(f"${channel.upper()}$ Mass [GeV]", fontsize=14)
        ax.set_ylabel("Events / Bin", fontsize=14)
        ax.set_yscale('log')
        ax.set_xlim(bins[0], bins[-1])
        
        min_y = max(0.1, np.min(data[data > 0]) * 0.5)
        max_y = np.max(data) * 5.0
        ax.set_ylim(min_y, max_y)
        
        ax.legend(fontsize=11, frameon=True, edgecolor='black', framealpha=0.9)
        ax.grid(True, which="both", linestyle="--", alpha=0.3)

    # Plot Ch1
    plot_marginal(axes[0], args.ch1, bins1, centers1, d1, c5p_1, cGP_1, expected_toys1)
    # Plot Ch2
    plot_marginal(axes[1], args.ch2, bins2, centers2, d2, c5p_2, cGP_2, expected_toys2)

    plt.suptitle(f"Copula Marginal Fidelity Validation with Gaussian Processes | Trigger: {args.trigger.upper()}", fontsize=18, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    os.makedirs(os.path.join(base_dir, "plots"), exist_ok=True)
    out_path = os.path.join(base_dir, "plots", f"gp_marginal_agreement_{args.trigger}_{args.ch1}_{args.ch2}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Successfully saved GP marginal agreement plot to: {out_path}")

if __name__ == "__main__":
    main()
