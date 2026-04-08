#!/usr/bin/env python3
import os
import sys
import json
import numpy as np
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from argparse import ArgumentParser
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

# Ensure paths are correct based on your repo structure
current_script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_script_dir)
if repo_root not in sys.path: sys.path.append(repo_root)

from src.config import ATLAS_BINS
from src.models import FiveParam

def fit_gp_background(centers, density, density_err, min_len_scale_log):
    """
    Fits the GP with a 0-mean function directly to the log-density.
    """
    mask = density > 0
    X_log = np.log(centers[mask]).reshape(-1, 1)
    
    # 0-mean function: GP learns the entire log-density directly
    y_target = np.log(density[mask])
    y_err_target = density_err[mask] / density[mask]

    kernel = C(1.0, (1e-3, 1e2)) * RBF(length_scale=max(min_len_scale_log * 2.0, 0.5), length_scale_bounds=(min_len_scale_log, 5.0)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=y_err_target**2, n_restarts_optimizer=5, normalize_y=False)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X_log, y_target)
        
    X_full_log = np.log(centers).reshape(-1, 1)
    y_pred_target = gp.predict(X_full_log)
    
    return np.exp(y_pred_target)

def main(args):
    os.makedirs("plots", exist_ok=True)
    mass_types = ["jj", "bb", "jb", "je", "jm", "jg", "be", "bm", "bg"]
    
    bkg_expectations = {}
    channel_info = {}

    print(f"Initializing Frozen GP (Zero-Mean) Baselines for {args.trigger.upper()}...")
    
    # 1. Load Data and Establish GP Background
    for m in mass_types:
        fitfile_nom = os.path.join("fits", f"fitme_p5_{args.trigger}_{m}.json")
        try:
            with open(fitfile_nom, "r") as j_nom:
                d_nom = json.load(j_nom)
                
            fmin_val, fmax_val = float(d_nom['fmin']), float(d_nom['fmax'])
            v_bins = ATLAS_BINS[(ATLAS_BINS >= fmin_val) & (ATLAS_BINS <= fmax_val)]
            c = (v_bins[:-1] + v_bins[1:]) / 2
            widths = np.diff(v_bins)
            
            # Use analytic fit as the raw proxy for the GP to smooth over
            counts_nom = FiveParam(args.cms, c, *[float(p) for p in d_nom['parameters']]) 
            
            if np.sum(counts_nom) > 0:
                channel_info[m] = {'centers': c, 'bins': v_bins, 'widths': widths}
                
                bkg_density = counts_nom / widths
                bkg_err_density = np.sqrt(np.maximum(counts_nom, 1.0)) / widths
                
                smoothed_density = fit_gp_background(c, bkg_density, bkg_err_density, args.min_len)
                bkg_expectations[m] = smoothed_density * widths
                
        except Exception as e:
            print(f"Skipping {m}: {e}")
            continue

    # 2. Load Raw Mass Matrix
    mass_path = os.path.join("data", f"masses_{args.trigger}.npz")
    f_mass = np.load(mass_path)
    mass_matrix, col_names_mass = f_mass['masses'], list(f_mass['columns'])
    N_events = len(mass_matrix)

    # 3. Load Copula, Compute Truncation Bounds & CDFs
    copula_path = os.path.join("data", f"copula_{args.trigger}.npz")
    f_cop = np.load(copula_path)
    matrix_cop, col_names_cop = f_cop['copula'], list(f_cop['columns'])
    
    cdfs = {m: np.cumsum(b) / np.sum(b) for m, b in bkg_expectations.items()}
    u_bounds = {}
    
    for m, b in bkg_expectations.items():
        idx = col_names_mass.index(f"M{m}")
        masses = mass_matrix[:, idx]
        valid_masses = masses[masses > 0] * args.cms
        
        fmin_val = channel_info[m]['bins'][0]
        fmax_val = channel_info[m]['bins'][-1]
        
        if len(valid_masses) > 0:
            u_min = np.sum(valid_masses < fmin_val) / len(valid_masses)
            u_max = np.sum(valid_masses <= fmax_val) / len(valid_masses)
        else:
            u_min, u_max = 0.0, 1.0
        u_bounds[m] = (u_min, u_max)

    # =========================================================================
    # PLOT 1: Single Toy Overlay
    # =========================================================================
    print("Generating spectra overlay plot...")
    fig, axes = plt.subplots(3, 3, figsize=(22, 18))
    axes = axes.flatten()
    
    # Pre-generate global toy seeds
    N_draw = np.random.poisson(N_events)
    
    # Bootstrap Sample
    sampled_rows_boot = mass_matrix[np.random.choice(N_events, size=N_draw, replace=True)]
    
    # Decorrelated Bootstrap Sample
    shuffled_matrix = np.copy(mass_matrix)
    for col_idx in range(shuffled_matrix.shape[1]):
        np.random.shuffle(shuffled_matrix[:, col_idx])
    sampled_rows_decorr = shuffled_matrix[np.random.choice(N_events, size=N_draw, replace=True)]
    
    # Copula Sample
    N_draw_cop = np.random.poisson(len(matrix_cop))
    sampled_rows_cop = matrix_cop[np.random.choice(len(matrix_cop), size=N_draw_cop, replace=True)]

    for i, m in enumerate(mass_types):
        if m not in bkg_expectations: continue
        ax = axes[i]
        b = bkg_expectations[m]
        bins = channel_info[m]['bins']
        centers = channel_info[m]['centers']

        # Method A: Naive Independent Poisson
        toy_naive = np.random.poisson(b)

        # Method B: Poisson Bootstrap
        idx_mass = col_names_mass.index(f"M{m}")
        valid_masses_boot = sampled_rows_boot[sampled_rows_boot[:, idx_mass] > 0, idx_mass] * args.cms
        toy_boot, _ = np.histogram(valid_masses_boot, bins=bins)

        # Method C: Decorrelated Bootstrap
        valid_masses_decorr = sampled_rows_decorr[sampled_rows_decorr[:, idx_mass] > 0, idx_mass] * args.cms
        toy_decorr, _ = np.histogram(valid_masses_decorr, bins=bins)

        # Method D: Truncated Empirical Copula
        idx_cop = col_names_cop.index(f"M{m}")
        u_raw = sampled_rows_cop[sampled_rows_cop[:, idx_cop] >= 0, idx_cop]
        u_min, u_max = u_bounds[m]
        
        u_in_window = u_raw[(u_raw >= u_min) & (u_raw <= u_max)]
        if len(u_in_window) == 0:
            toy_copula = np.zeros(len(b), dtype=int)
        else:
            u_jittered = u_in_window + np.random.uniform(-0.0002, 0.0002, size=len(u_in_window))
            u_trunc = (u_jittered - u_min) / (u_max - u_min)
            u_trunc = np.abs(u_trunc)
            u_trunc = np.where(u_trunc >= 1.0, 1.99999 - u_trunc, u_trunc)
            toy_copula = np.bincount(np.searchsorted(cdfs[m], u_trunc), minlength=len(b))

        # Plotting
        ax.plot(centers, b, label='Expected Bkg (GP Zero)', color='black', linestyle='--', linewidth=2, zorder=5)
        
        ax.errorbar(centers, toy_naive, yerr=np.sqrt(toy_naive), fmt='o', color='blue', alpha=0.5, 
                    label='Naive Poisson', markersize=5, zorder=1)
        ax.errorbar(centers, toy_boot, yerr=np.sqrt(toy_boot), fmt='s', color='red', alpha=0.5, 
                    label='Poisson Bootstrap', markersize=5, zorder=2)
        ax.errorbar(centers, toy_decorr, yerr=np.sqrt(toy_decorr), fmt='D', color='orange', alpha=0.5, 
                    label='Decorrelated Bootstrap', markersize=5, zorder=3)
        ax.errorbar(centers, toy_copula, yerr=np.sqrt(toy_copula), fmt='^', color='green', alpha=0.7, 
                    label='Truncated Copula', markersize=5, zorder=4)
        
        ax.set_title(f"Channel: {m}", fontsize=14)
        ax.set_yscale('log')
        ax.set_xlabel("Mass [GeV]", fontsize=12)
        ax.set_ylabel("Events", fontsize=12)
        ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(f"plots/spectra_overlay_GPzero_{args.trigger}.png", dpi=300)
    plt.close()

    # =========================================================================
    # PLOT 2: Cross-Channel Covariance/Correlation from Data
    # =========================================================================
    print("Generating correlation matrix (Empirical Copula/Bootstrap Structure)...")
    N_ensemble = 1000
    yields = {m: np.zeros(N_ensemble) for m in bkg_expectations.keys()}

    for i in range(N_ensemble):
        N_d = np.random.poisson(N_events)
        ens_rows = mass_matrix[np.random.choice(N_events, size=N_d, replace=True)]
        for m in bkg_expectations.keys():
            idx = col_names_mass.index(f"M{m}")
            masses = ens_rows[:, idx]
            # Must strictly count events inside the fit window for accurate correlation mapping
            fmin_val, fmax_val = channel_info[m]['bins'][0], channel_info[m]['bins'][-1]
            valid = masses[(masses * args.cms >= fmin_val) & (masses * args.cms <= fmax_val)]
            yields[m][i] = len(valid)

    df_yields = pd.DataFrame(yields)
    corr_matrix = df_yields.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=0, vmax=1, 
                square=True, linewidths=.5, cbar_kws={"shrink": .8})
    plt.title(f"Cross-Channel Yield Correlation\n(Evaluated inside fit windows, 1000 Toys)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"plots/correlation_matrix_GPzero_{args.trigger}.png", dpi=300)
    plt.close()

    print("Done! Plots saved to the 'plots/' directory.")

if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--trigger', required=True)
    p.add_argument('--cms', type=float, default=13000.)
    p.add_argument('--min-len', type=float, default=0.15)
    main(p.parse_args())
