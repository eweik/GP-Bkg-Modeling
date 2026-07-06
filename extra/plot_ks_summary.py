#!/usr/bin/env python3
import os
import sys
import json
import uproot
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, kstest
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# ATLAS Variable Bins
ATLAS_BINS = np.array([
    99,112,125,138,151,164,177,190, 203, 216, 229, 243, 257, 272, 287, 303, 319, 335, 352, 369, 
    387, 405, 424, 443, 462, 482, 502, 523, 544, 566, 588, 611, 634, 657, 681, 705, 730, 755, 
    781, 807, 834, 861, 889, 917, 946, 976, 1006, 1037, 1068, 1100, 1133, 1166, 1200, 1234, 1269, 
    1305, 1341, 1378, 1416, 1454, 1493, 1533, 1573, 1614, 1656, 1698, 1741, 1785, 1830, 1875, 1921, 
    1968, 2016, 2065, 2114, 2164, 2215, 2267, 2320, 2374, 2429, 2485, 2542, 2600, 2659, 2719, 2780, 
    2842, 2905, 2969, 3034, 3100, 3167, 3235, 3305, 3376, 3448, 3521, 3596, 3672, 3749, 3827, 3907, 
    3988, 4070, 4154, 4239, 4326, 4414, 4504, 4595, 4688, 4782, 4878, 4975, 5074, 5175, 5277, 5381, 
    5487, 5595, 5705, 5817, 5931, 6047, 6165, 6285, 6407, 6531, 6658, 6787, 6918, 7052, 7188, 7326, 
    7467, 7610, 7756, 7904, 8055, 8208, 8364, 8523, 8685, 8850, 9019, 9191, 9366, 9544, 9726, 9911, 
    10100, 10292, 10488, 10688, 10892, 11100, 11312, 11528, 11748, 11972, 12200, 12432, 12669, 12910, 13156
])

def ParametricFit(Ecm, x_center, p):
    p_safe = np.zeros(5)
    for i in range(min(len(p), 5)): p_safe[i] = p[i]
    x = x_center / Ecm
    nlog = np.log(x)
    return p_safe[0] * np.power(np.maximum(1.0 - x, 1e-10), p_safe[1]) * np.power(x, (p_safe[2] + p_safe[3] * nlog + p_safe[4] * nlog * nlog))

def get_gp_fit(centers, density, density_err, min_len_scale_log=0.15, max_len_scale_log=5.0):
    """Calculates the zero-mean GP fit directly on log density."""
    mask = density > 0
    if np.sum(mask) < 5: return density, np.zeros_like(density), False

    X_log = np.log(centers[mask]).reshape(-1, 1)
    y_target = np.log(density[mask])
    y_err_target = density_err[mask] / density[mask]

    kernel = C(1.0, (1e-3, 1e2)) * RBF(length_scale=min_len_scale_log, length_scale_bounds=(min_len_scale_log, max_len_scale_log))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=y_err_target**2, n_restarts_optimizer=5, normalize_y=False)
    
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(X_log, y_target)
            
        X_full_log = np.log(centers).reshape(-1, 1)
        y_pred_target, y_std = gp.predict(X_full_log, return_std=True)
        
        return np.exp(y_pred_target), np.exp(y_pred_target) * y_std, True
    except Exception:
        return density, np.zeros_like(density), False

def get_atlas_binned_data(hist):
    counts, edges = hist.to_numpy()
    centers_raw = (edges[:-1] + edges[1:]) / 2
    binned_counts, _ = np.histogram(centers_raw, bins=ATLAS_BINS, weights=counts)
    return binned_counts

def main():
    parser = argparse.ArgumentParser(description="Aggregate and Plot KS Probabilities (Zero-Mean GP vs 5-Param)")
    parser.add_argument("--root-dir", type=str, default="/afs/cern.ch/user/e/edweik/private/new_ad_files")
    parser.add_argument("--fits-dir", type=str, default="/afs/cern.ch/user/e/edweik/private/GlobalSignificanceSearch/fits")
    parser.add_argument("--sqrts", type=float, default=13000.0)
    parser.add_argument("--min-len", type=float, default=0.15)
    parser.add_argument("--max-len", type=float, default=5.0)
    args = parser.parse_args()

    out_dir = "plots/ks_summaries"
    os.makedirs(out_dir, exist_ok=True)

    channels = ["jj", "jb", "bb", "be", "bm", "bg", "je", "jm", "jg"]
    triggers = [f"t{i}" for i in range(1, 8)]

    results = []

    print("Extracting KS p-values across all triggers and channels...")

    for t in triggers:
        root_file_path = os.path.join(args.root_dir, f"data1percent_{t}_HAE_RUN23_nominal_10PB.root")
        
        try:
            root_file = uproot.open(root_file_path)
        except Exception:
            continue

        for ch in channels:
            hist_name = f"M{ch}_data1percent"
            json_file_nom = os.path.join(args.fits_dir, f"fitme_p5_{t}_{ch}.json")

            if hist_name not in root_file or not os.path.exists(json_file_nom):
                continue

            with open(json_file_nom, "r") as f:
                d_nom = json.load(f)
            
            fmin, fmax = float(d_nom['fmin']), float(d_nom['fmax'])
            params_nom = d_nom['parameters']

            full_counts = get_atlas_binned_data(root_file[hist_name])
            centers_full = (ATLAS_BINS[:-1] + ATLAS_BINS[1:]) / 2
            widths_full = np.diff(ATLAS_BINS)

            mask = (centers_full >= fmin) & (centers_full <= fmax)
            c_fit, w_fit, data_counts = centers_full[mask], widths_full[mask], full_counts[mask]

            data_density = data_counts / w_fit
            data_err_density = np.sqrt(np.maximum(data_counts, 1.0)) / w_fit

            fit_leg_counts = ParametricFit(args.sqrts, c_fit, params_nom)
            fit_leg_density = fit_leg_counts / w_fit

            gp_density, gp_err, ok = get_gp_fit(
                c_fit, data_density, data_err_density, 
                min_len_scale_log=args.min_len, max_len_scale_log=args.max_len
            )

            if not ok: continue

            # Calculate pulls
            pulls_leg = np.where(data_err_density > 0, (data_density - fit_leg_density) / data_err_density, 0)
            pulls_gp = np.where(data_err_density > 0, (data_density - gp_density) / data_err_density, 0)

            v_leg = pulls_leg[data_counts > 0]
            v_gp = pulls_gp[data_counts > 0]

            if len(v_leg) > 0 and len(v_gp) > 0:
                _, pval_leg = kstest(v_leg, 'norm')
                _, pval_gp = kstest(v_gp, 'norm')
                
                results.append({
                    'label': f"{t.upper()}_{ch.upper()}",
                    'pval_nom': pval_leg,
                    'pval_gp': pval_gp
                })
                sys.stdout.write(f"\rProcessed {t.upper()}_{ch.upper()}...")
                sys.stdout.flush()

    print("\n\nGenerating Summary Plots...")
    
    if not results:
        print("No valid data points found. Exiting.")
        sys.exit(1)

    p_nom = [r['pval_nom'] for r in results]
    p_gp = [r['pval_gp'] for r in results]

    # ==========================================
    # PLOT 1: 1D Histograms
    # ==========================================
    fig_hist, ax_hist = plt.subplots(figsize=(8, 6))
    bins = np.linspace(0, 1.0, 21)
    
    ax_hist.hist(p_nom, bins=bins, histtype='stepfilled', alpha=0.4, color='red', label='Nominal 5-Param')
    ax_hist.hist(p_nom, bins=bins, histtype='step', color='red', linewidth=1.5)
    
    ax_hist.hist(p_gp, bins=bins, histtype='stepfilled', alpha=0.4, color='blue', label='Gaussian Process')
    ax_hist.hist(p_gp, bins=bins, histtype='step', color='blue', linewidth=1.5)

    ax_hist.set_title(f"KS Test p-value Distributions (N={len(results)})", fontsize=14, fontweight='bold')
    ax_hist.set_xlabel("KS p-value (against standard normal)", fontsize=12)
    ax_hist.set_ylabel("Number of Channels", fontsize=12)
    ax_hist.set_xlim(0, 1.0)
    ax_hist.legend(loc='upper right')
    ax_hist.grid(True, linestyle='--', alpha=0.5)

    out_file_hist = os.path.join(out_dir, "ks_probability_histograms_zeromean.png")
    fig_hist.tight_layout()
    fig_hist.savefig(out_file_hist, dpi=300, bbox_inches='tight')
    print(f"Saved histogram plot to {out_file_hist}")
    plt.close(fig_hist)

    # ==========================================
    # PLOT 2: Scatter Plot
    # ==========================================
    fig_scat, ax_scat = plt.subplots(figsize=(8, 8))
    ax_scat.scatter(p_nom, p_gp, color='purple', alpha=0.7, edgecolor='k', s=60, zorder=3)
    
    for r in results:
        # Label points where either model performs poorly (< 0.05)
        if r['pval_gp'] < 0.05 or r['pval_nom'] < 0.05:
            ax_scat.annotate(r['label'], (r['pval_nom'], r['pval_gp']), 
                             xytext=(4,4), textcoords='offset points', fontsize=8, alpha=0.7)

    # Diagonal line y=x indicating equal performance
    ax_scat.plot([-0.05, 1.05], [-0.05, 1.05], 'k--', zorder=1, alpha=0.6, label='Equal p-value ($y=x$)')
    
    ax_scat.set_title(f"GP vs 5-Param KS p-values\nLength Scale bounds: $\ell \in [{args.min_len}, {args.max_len}]$", fontsize=14, fontweight='bold')
    ax_scat.set_xlabel("Nominal 5-Param KS p-value", fontsize=12)
    ax_scat.set_ylabel("GP KS p-value", fontsize=12)
    ax_scat.set_xlim(-0.05, 1.05)
    ax_scat.set_ylim(-0.05, 1.05)
    
    # Draw standard significance threshold (p=0.05)
    ax_scat.axvline(0.05, color='red', linestyle=':', alpha=0.6, label='p = 0.05 threshold')
    ax_scat.axhline(0.05, color='blue', linestyle=':', alpha=0.6)
    
    ax_scat.legend(loc='lower right')
    ax_scat.grid(True, linestyle='--', alpha=0.5)

    out_file_scat = os.path.join(out_dir, "ks_probability_scatter_zeromean.png")
    fig_scat.tight_layout()
    fig_scat.savefig(out_file_scat, dpi=300, bbox_inches='tight')
    print(f"Saved scatter plot to {out_file_scat}")
    plt.close(fig_scat)

if __name__ == "__main__":
    main()
