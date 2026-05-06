#!/usr/bin/env python3
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import argparse
import os
import json
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Setup paths to import local modules
current_script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_script_dir)
if repo_root not in sys.path: sys.path.append(repo_root)
if os.getcwd() not in sys.path: sys.path.append(os.getcwd())

from src.config import ATLAS_BINS
from src.models import FiveParam
from src.stats import fast_bumphunter_stat

def fit_gp_background(centers, density, density_err, parametric_density, min_len_scale_log, max_len_scale_log, gp_mean_type):
    """Fits the GP to establish the expected background."""
    mask = density > 0
    X_log = np.log(centers[mask]).reshape(-1, 1)
    
    if gp_mean_type == 'zero':
        y_target = np.log(density[mask])
    else:
        y_target = np.log(density[mask] / np.maximum(parametric_density[mask], 1e-15))

    y_err_target = density_err[mask] / density[mask]

    kernel = C(1.0, (1e-3, 1e2)) * RBF(
        length_scale=max(min_len_scale_log, 1e-4),
        length_scale_bounds=(min_len_scale_log, max_len_scale_log) 
    )
    gp = GaussianProcessRegressor(kernel=kernel, alpha=y_err_target**2, n_restarts_optimizer=5, normalize_y=False)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X_log, y_target)
        
    X_full_log = np.log(centers).reshape(-1, 1)
    y_pred_target = gp.predict(X_full_log)
    
    if gp_mean_type == 'zero':
        return np.exp(y_pred_target)
    else:
        return parametric_density * np.exp(y_pred_target)


def main():
    parser = argparse.ArgumentParser(description="Calculate Trigger-Wide Global Significance (GP Model) with Empirical Data.")
    parser.add_argument("--trigger", type=str, default="t2", help="Target trigger to evaluate")
    parser.add_argument("--ExpectedLocalZvalue", type=float, default=5.0, help="Target local significance")
    parser.add_argument('--cms', type=float, default=13000., help="Center of mass energy")
    
    # GP specific arguments needed to recreate the Null Hypothesis
    parser.add_argument('--min-len', type=float, default=0.15)
    parser.add_argument('--max-len', type=float, default=5.0) 
    parser.add_argument('--gp-mean', choices=['5param', 'zero'], default='zero')
    args = parser.parse_args()

    trigger = args.trigger.lower()
    target_Z = args.ExpectedLocalZvalue
    
    methods = ["naive", "copula", "poisson_event"]
    colors = {"naive": "red", "copula": "orange", "poisson_event": "green"}
    method_label_map = {"naive": "Independent", "copula": "Copula", "poisson_event": "Poisson Bootstrap"}
    
    os.makedirs("plots", exist_ok=True)
    base_dir = os.getcwd() if os.path.exists("data") else repo_root

    print(f"\n{'='*65}")
    print(f" TRIGGER-WIDE GLOBAL SIGNIFICANCE ({trigger.upper()}) | GP BACKGROUND")
    print(f"{'='*65}")

    # --- 1. CALCULATE OBSERVED EMPIRICAL DATA STATISTIC (GP NULL) ---
    t_data_max = 0.0
    z_data_local = -np.inf
    mass_path = os.path.join(base_dir, "data", f"masses_{trigger}.npz")

    if os.path.exists(mass_path):
        print("\nFitting Gaussian Process to raw data to find observed empirical significance...")
        f_mass = np.load(mass_path)
        mass_matrix = f_mass['masses']
        col_names = list(f_mass['columns'])
        mass_types = ["jj", "bb", "jb", "je", "jm", "jg", "be", "bm", "bg"]

        for m in mass_types:
            fitfile = os.path.join(base_dir, "fits", f"fitme_p5_{trigger}_{m}.json")
            if not os.path.exists(fitfile): continue

            with open(fitfile, "r") as f: d_nom = json.load(f)

            fmin_val, fmax_val = float(d_nom['fmin']), float(d_nom['fmax'])
            v_bins = ATLAS_BINS[(ATLAS_BINS >= fmin_val) & (ATLAS_BINS <= fmax_val)]
            c = (v_bins[:-1] + v_bins[1:]) / 2
            widths = np.diff(v_bins)

            idx = col_names.index(f"M{m}")
            valid_masses = mass_matrix[mass_matrix[:, idx] > 0, idx] * args.cms
            data_counts, _ = np.histogram(valid_masses, bins=v_bins)

            if np.sum(data_counts) < 50: continue

            # Fit the GP to establish the expected null
            bkg_density = data_counts / widths
            bkg_err_density = np.sqrt(np.maximum(data_counts, 1.0)) / widths
            parametric_density = FiveParam(args.cms, c, *[float(p) for p in d_nom['parameters']]) / widths
            
            smoothed_density = fit_gp_background(c, bkg_density, bkg_err_density, parametric_density, args.min_len, args.max_len, args.gp_mean)
            bkg_counts = smoothed_density * widths

            # Evaluate BumpHunter against GP Null
            t_ch = fast_bumphunter_stat(data_counts, bkg_counts)
            t_data_max = max(t_data_max, t_ch)

        if t_data_max > 0:
            p_data_local = np.exp(-t_data_max)
            p_data_local = np.clip(p_data_local, 1e-300, 0.999999)
            z_data_local = stats.norm.isf(p_data_local)
            print(f"Observed Maximum Local Test Statistic (t): {t_data_max:.3f}")
            print(f"Observed Maximum Local Z-score: {z_data_local:.3f}σ")
        else:
            print("Data perfectly matches background (t=0). Local Z is negligible.")
    else:
        print(f"Warning: Mass matrix not found at {mass_path}. Cannot compute empirical data point.")

    plt.figure(figsize=(11, 7))

    # --- 2. EVALUATE TOY METHODS ---
    for method in methods:
        # Load the GP specifically-generated toys
        file_list = glob.glob(f"results/global_stat_{trigger}_{method}_{args.gp_mean}_*.npy")
        if not file_list:
            file_list = glob.glob(f"results/merged15/final_{trigger}_{method}.npy") # Fallback to merged
            
        if not file_list:
            print(f"[{method.upper()}] Missing data for {trigger}. Skipping.")
            continue
            
        arr = np.concatenate([np.load(f) for f in file_list])
        trigger_t_max = arr[np.isfinite(arr)]
        n_toys = len(trigger_t_max)
        
        if n_toys == 0:
            continue
            
        p_local_dist = np.exp(-trigger_t_max)
        p_local_dist = np.clip(p_local_dist, 1e-300, 0.999999)
        z_local_dist = stats.norm.isf(p_local_dist)

        # Target Statistics
        NrFound = np.sum(z_local_dist >= target_Z)
        p_global = NrFound / n_toys
        Z_global = stats.norm.isf(p_global) if p_global > 0 else np.inf

        # Empirical Observed Stats
        emp_p_global = np.sum(trigger_t_max >= t_data_max) / n_toys
        emp_Z_global = stats.norm.isf(emp_p_global) if emp_p_global > 0 else np.inf

        print(f"\n###### RESULT: {method.upper()} ######")
        print(f" Number of pseudo-experiments = {n_toys}")
        print(f" --- Target (Local Z = {target_Z}) ---")
        print(f" Toys >= Target: {NrFound}")
        if p_global > 0:
            print(f" Expected Global p-value = {p_global:.2e} (Global Z = {Z_global:.2f})")
        else:
            print(f" Expected Global p-value = < {1/n_toys:.2e} (Need more toys)")

        if t_data_max > 0:
            emp_toys_found = np.sum(trigger_t_max >= t_data_max)
            print(f" --- Empirical Data (Local Z = {z_data_local:.2f}) ---")
            print(f" Toys >= Observed Data: {emp_toys_found}")
            if emp_p_global > 0:
                print(f" Empirical Global p-value = {emp_p_global:.2e}  (Global Z = {emp_Z_global:.2f})")
            else:
                print(f" Empirical Global p-value = < {1/n_toys:.2e} (Exceeds toy limits)")
            
        # Plot Survival Curve
        z_local_sorted = np.sort(z_local_dist)[::-1]
        ranks = np.arange(1, n_toys + 1)
        p_global_curve = ranks / n_toys
        z_global_curve = stats.norm.isf(p_global_curve)

        valid = (z_global_curve > -10) & np.isfinite(z_global_curve)
        method_label = method_label_map.get(method, method.capitalize())
        c_color = colors.get(method, 'black')
        
        plt.plot(z_local_sorted[valid], z_global_curve[valid],
                 label=f"{method_label} (N={n_toys})", color=c_color, lw=2)

        # Plot Empirical Intersection Point
        if z_data_local > 0 and emp_Z_global < np.inf:
            plt.plot(z_data_local, emp_Z_global, marker='o', color=c_color, markersize=8, zorder=5)

    # --- 3. FORMAT THE PLOT ---
    plt.title(f"Trigger-Wide Global Significance vs. BumpHunter Significance\nTrigger: {trigger.upper()} | GP Background Model", fontsize=15, fontweight='bold')
    plt.xlabel(f"Highest Observed Local Significance in {trigger.upper()} ($Z_{{BH}}$)", fontsize=12)
    plt.ylabel("Trigger-Wide Global Significance ($Z_{global}$)", fontsize=12)
    
    plt.axhline(3, color='grey', linestyle='--', alpha=0.7, label='3σ Global Evidence')
    plt.axhline(5, color='black', linestyle=':', alpha=0.7, label='5σ Global Discovery')
    
    # Empirical Data Line
    if z_data_local > 0:
        plt.axvline(z_data_local, color='magenta', linestyle='-.', lw=2, alpha=0.8, label=f'Observed Data ($Z_{{local}}={z_data_local:.2f}$)')
    
    lims = [max(0, plt.xlim()[0]), min(8, plt.xlim()[1])]
    plt.plot(lims, lims, 'k--', alpha=0.3, label="No LEE ($Z_{global} = Z_{BH}$)")

    plt.legend(loc="lower right", fontsize=10, framealpha=0.9, edgecolor='black')
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    plot_out = f"plots/Trigger_Wide_Global_Z_{trigger}_GP_Empirical.png"
    plt.savefig(plot_out, dpi=300)
    print(f"\n{'-'*65}\nPlot saved to {plot_out}\n")

if __name__ == "__main__":
    main()
