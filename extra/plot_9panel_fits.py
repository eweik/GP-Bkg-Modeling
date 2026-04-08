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

# Adjust paths
current_script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_script_dir)
if repo_root not in sys.path: sys.path.append(repo_root)
if os.getcwd() not in sys.path: sys.path.append(os.getcwd())

from src.config import ATLAS_BINS
from src.models import FiveParam

def get_gp_fit(centers, density, density_err, min_len_scale_log=0.15):
    """Calculates the zero-mean GP fit."""
    mask = density > 0
    X_log = np.log(centers[mask]).reshape(-1, 1)
    y_target = np.log(density[mask])
    y_err_target = density_err[mask] / density[mask]

    # Cleaned kernel: Removed WhiteKernel, relying strictly on alpha (data errors)
    #kernel = C(1.0, (1e-3, 1e2)) * RBF(length_scale=0.3, length_scale_bounds=(min_len_scale_log, 5.0))
    kernel = C(1.0, (1e-3, 1e2)) * RBF(length_scale=min_len_scale_log, length_scale_bounds=(min_len_scale_log, 5.0))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=y_err_target**2, n_restarts_optimizer=5, normalize_y=False)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X_log, y_target)
        
    X_full_log = np.log(centers).reshape(-1, 1)
    y_pred_target, y_std = gp.predict(X_full_log, return_std=True)
    
    # Return expected density and the 1-sigma uncertainty band
    return np.exp(y_pred_target), np.exp(y_pred_target) * y_std

def main():
    parser = argparse.ArgumentParser(description="Plot 9-Panel Diagnostic: GP vs 5-Param")
    parser.add_argument('--trigger', type=str, required=True, help="Trigger name (e.g., t2)")
    parser.add_argument('--cms', type=float, default=13000., help="Center of mass energy")
    parser.add_argument('--min-len', type=float, default=0.15, help="Minimum log-length scale for GP (e.g., 0.15 for ~16% mass window)")
    args = parser.parse_args()

    trigger = args.trigger.lower()
    cms = args.cms
    mass_types = ["jj", "bb", "jb", "je", "jm", "jg", "be", "bm", "bg"]
    
    base_dir = os.getcwd() if os.path.exists("data") else repo_root
    mass_path = os.path.join(base_dir, "data", f"masses_{trigger}.npz")
    
    if not os.path.exists(mass_path):
        print(f"Error: Mass data not found at {mass_path}")
        sys.exit(1)
        
    f_mass = np.load(mass_path)
    mass_matrix, col_names = f_mass['masses'], list(f_mass['columns'])

    print(f"Generating 9-panel diagnostic for Trigger: {trigger.upper()}...")

    fig = plt.figure(figsize=(24, 18))
    # Create a 3x3 grid of subplots, each with a main panel and a ratio panel
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.25)

    for i, m in enumerate(mass_types):
        row = i // 3
        col = i % 3
        
        # Sub-gridspec for main + ratio panel
        inner_gs = gs[row, col].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
        ax_main = fig.add_subplot(inner_gs[0])
        ax_ratio = fig.add_subplot(inner_gs[1], sharex=ax_main)

        fitfile = os.path.join(base_dir, "fits", f"fitme_p5_{trigger}_{m}.json")
        if not os.path.exists(fitfile):
            ax_main.text(0.5, 0.5, f"No fit found for {m.upper()}", ha='center', va='center')
            continue

        with open(fitfile, "r") as j_nom:
            d_nom = json.load(j_nom)

        fmin_val, fmax_val = float(d_nom['fmin']), float(d_nom['fmax'])
        v_bins = ATLAS_BINS[(ATLAS_BINS >= fmin_val) & (ATLAS_BINS <= fmax_val)]
        c = (v_bins[:-1] + v_bins[1:]) / 2
        widths = np.diff(v_bins)

        # 1. Analytic 5-Param Fit
        counts_5p = FiveParam(cms, c, *[float(p) for p in d_nom['parameters']])
        density_5p = counts_5p / widths

        # 2. Raw Data Extraction
        idx = col_names.index(f"M{m}")
        valid_masses = mass_matrix[mass_matrix[:, idx] > 0, idx] * cms
        raw_counts, _ = np.histogram(valid_masses, bins=v_bins)
        
        bkg_density = raw_counts / widths
        bkg_err_density = np.sqrt(np.maximum(raw_counts, 1.0)) / widths

        # 3. GP Fit (Zero Mean)
        gp_density, gp_err = get_gp_fit(c, bkg_density, bkg_err_density, min_len_scale_log=args.min_len)

        # --- PLOTTING MAIN PANEL ---
        # Data points
        valid = raw_counts > 0
        ax_main.errorbar(c[valid], bkg_density[valid], yerr=bkg_err_density[valid], 
                         fmt='ko', markersize=4, label='Raw Data', zorder=5)
        
        # 5-Param Line
        ax_main.plot(c, density_5p, color='dodgerblue', linestyle='--', linewidth=2, label='5-Param Fit')
        
        # GP Line & Uncertainty
        ax_main.plot(c, gp_density, color='red', linewidth=2, label='GP (Zero Mean)')
        ax_main.fill_between(c, gp_density - gp_err, gp_density + gp_err, color='red', alpha=0.2)

        ax_main.set_yscale('log')
        ax_main.set_ylabel('Events / GeV', fontsize=12)
        ax_main.set_title(f"Channel: {m.upper()}", fontsize=14, fontweight='bold')
        ax_main.tick_params(labelbottom=False)
        if i == 0: ax_main.legend(fontsize=12)

        # --- PLOTTING RATIO PANEL ---
        # Ratio: Data / Fit
        ratio_5p = np.where(density_5p > 0, bkg_density / density_5p, np.nan)
        ratio_gp = np.where(gp_density > 0, bkg_density / gp_density, np.nan)
        ratio_err = np.where(bkg_density > 0, bkg_err_density / bkg_density, 0)

        ax_ratio.errorbar(c[valid], ratio_5p[valid], yerr=ratio_err[valid]*ratio_5p[valid], 
                          fmt='o', color='dodgerblue', markersize=3, alpha=0.6)
        ax_ratio.errorbar(c[valid], ratio_gp[valid], yerr=ratio_err[valid]*ratio_gp[valid], 
                          fmt='o', color='red', markersize=3, alpha=0.8)
        
        ax_ratio.axhline(1.0, color='black', linestyle='--', linewidth=1)
        ax_ratio.set_ylim(0.5, 1.5)
        ax_ratio.set_ylabel('Data / Model', fontsize=10)
        ax_ratio.set_xlabel('Mass [GeV]', fontsize=12)
        
        # Only show y-ticks for the outer plots to keep it clean
        if col != 0: 
            ax_main.tick_params(labelleft=False)
            ax_ratio.tick_params(labelleft=False)

    # --- THE FIX IS HERE ---
    # 1. Added the global title without forcing the 'y' parameter
    plt.suptitle(f"Background Modeling Comparison: 5-Parameter vs Gaussian Process\nTrigger: {trigger.upper()} | Min Length Scale $\ell \geq {args.min_len}$", fontsize=22, fontweight='bold')
    
    # 2. Force tight_layout to leave the top 4% of the canvas completely empty for the suptitle
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    
    os.makedirs(os.path.join(base_dir, "plots"), exist_ok=True)
    out_path = os.path.join(base_dir, "plots", f"gp_vs_5param_9panel_{trigger}.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved 9-panel diagnostic plot to {out_path}")

if __name__ == "__main__":
    main()
