#!/usr/bin/env python3
import os
import sys
import json
import uproot
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

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
    for i in range(min(len(p), 5)):
        p_safe[i] = p[i]
    x = x_center / Ecm
    nlog = np.log(x)
    return p_safe[0] * np.power(np.maximum(1.0 - x, 1e-10), p_safe[1]) * np.power(x, (p_safe[2] + p_safe[3] * nlog + p_safe[4] * nlog * nlog))

def fit_gp_density(centers, density, density_err, min_len_scale=300.0):
    mask = density > 0
    if np.sum(mask) < 5:
        return density, False
    X = centers[mask].reshape(-1, 1)
    y = density[mask]
    y_err = density_err[mask]
    y_log = np.log(y)
    y_err_log = y_err / y

    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=max(min_len_scale * 2, 1000.0), length_scale_bounds=(min_len_scale, 1e4)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=y_err_log**2, n_restarts_optimizer=5, normalize_y=True)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(X, y_log)
        y_pred_log = gp.predict(centers.reshape(-1, 1))
        return np.exp(y_pred_log), True
    except Exception:
        return density, False

def get_atlas_binned_data(hist):
    counts, edges = hist.to_numpy()
    centers_raw = (edges[:-1] + edges[1:]) / 2
    binned_counts, _ = np.histogram(centers_raw, bins=ATLAS_BINS, weights=counts)
    return binned_counts

def plot_advanced_pull_diagnostics(pulls_legacy, pulls_gp, fit_name, out_base_path, title_prefix=""):
    leg_clean = pulls_legacy[(pulls_legacy != 0.0) & np.isfinite(pulls_legacy)]
    gp_clean = pulls_gp[(pulls_gp != 0.0) & np.isfinite(pulls_gp)]
    
    if len(leg_clean) < 10 or len(gp_clean) < 10:
        return False, None, f"Not enough valid pulls (< 10) for KS test. (Legacy: {len(leg_clean)}, GP: {len(gp_clean)})"

    mu_leg, std_leg = np.mean(leg_clean), np.std(leg_clean)
    mu_gp, std_gp = np.mean(gp_clean), np.std(gp_clean)
    
    _, ks_pval_leg = stats.kstest(leg_clean, 'norm', args=(0, 1))
    _, ks_pval_gp = stats.kstest(gp_clean, 'norm', args=(0, 1))

    stats_dict = {
        'mu_leg': mu_leg, 'std_leg': std_leg, 'ks_leg': ks_pval_leg,
        'mu_gp': mu_gp, 'std_gp': std_gp, 'ks_gp': ks_pval_gp
    }

    x_pdf = np.linspace(-4, 4, 100)
    bins = np.linspace(-4, 4, 25)

    # --- PLOT 1: Histograms and Empirical Fits ---
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    
    label_leg = f"Legacy {fit_name}\n$\mu={mu_leg:.2f}, \sigma={std_leg:.2f}$\nKS p-val = {ks_pval_leg:.3e}"
    ax1.hist(leg_clean, bins=bins, color='red', alpha=0.4, density=True, label=label_leg)
    
    label_gp = f"Gaussian Process\n$\mu={mu_gp:.2f}, \sigma={std_gp:.2f}$\nKS p-val = {ks_pval_gp:.3e}"
    ax1.hist(gp_clean, bins=bins, color='blue', alpha=0.4, density=True, label=label_gp)

    # Theoretical vs Empirical Normal Curves
    ax1.plot(x_pdf, stats.norm.pdf(x_pdf, 0, 1), 'k--', lw=2, label="Theoretical $\mathcal{N}(0,1)$")
    ax1.plot(x_pdf, stats.norm.pdf(x_pdf, mu_leg, std_leg), 'r:', lw=2, label=f"Empirical Fit ({fit_name})")
    ax1.plot(x_pdf, stats.norm.pdf(x_pdf, mu_gp, std_gp), 'b:', lw=2, label="Empirical Fit (GP)")
    
    ax1.set_title(f"{title_prefix}\nPull Density & Normality Test", fontsize=14, fontweight='bold')
    ax1.set_xlabel("Pull Value", fontsize=12)
    ax1.set_ylabel("Density", fontsize=12)
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{out_base_path}_Hist.png", dpi=300)
    plt.close(fig1)

    # --- PLOT 2: Q-Q Plot ---
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    
    osm_leg, osr_leg = stats.probplot(leg_clean, dist="norm", fit=False)
    osm_gp, osr_gp = stats.probplot(gp_clean, dist="norm", fit=False)

    ax2.plot(osm_leg, osr_leg, 'ro', alpha=0.6, label=f"Legacy {fit_name}")
    ax2.plot(osm_gp, osr_gp, 'bo', alpha=0.6, label="Gaussian Process")
    
    lims = [-4, 4]
    ax2.plot(lims, lims, 'k--', lw=2, label="Perfect Gaussian")
    
    ax2.set_title(f"{title_prefix}\nQ-Q Plot (Tail Assessment)", fontsize=14, fontweight='bold')
    ax2.set_xlabel("Theoretical Quantiles $\mathcal{N}(0,1)$", fontsize=12)
    ax2.set_ylabel("Empirical Pull Quantiles", fontsize=12)
    ax2.set_xlim(-4, 4)
    ax2.set_ylim(-4, 4)
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{out_base_path}_QQ.png", dpi=300)
    plt.close(fig2)
    
    return True, stats_dict, ""

def main():
    parser = argparse.ArgumentParser(description="Advanced Pull Diagnostics for Background Models.")
    parser.add_argument("--root-dir", type=str, required=True)
    parser.add_argument("--fits-dir", type=str, required=True)
    parser.add_argument("--sqrts", type=float, default=13000.0)
    args = parser.parse_args()

    out_dir_hist = "plots/pull_diagnostics/histograms"
    out_dir_qq = "plots/pull_diagnostics/qq_plots"
    os.makedirs(out_dir_hist, exist_ok=True)
    os.makedirs(out_dir_qq, exist_ok=True)

    channels = ["jj", "jb", "bb", "be", "bm", "bg", "je", "jm", "jg"]
    triggers = [f"t{i}" for i in range(1, 8)]

    completed, skipped = 0, 0
    results_table = []
    
    print("\nStarting Pull Diagnostics Sweep...")
    print("-" * 50)

    for t in triggers:
        root_path = os.path.join(args.root_dir, f"data1percent_{t}_HAE_RUN23_nominal_10PB.root")
        if not os.path.exists(root_path):
            print(f"[SKIP] {t.upper()}: ROOT file missing entirely.")
            skipped += len(channels)
            continue
            
        try:
            root_file = uproot.open(root_path)
        except Exception as e:
            print(f"[SKIP] {t.upper()}: Could not open ROOT file ({e}).")
            skipped += len(channels)
            continue

        for ch in channels:
            hist_name = f"M{ch}_data1percent"
            json_file = os.path.join(args.fits_dir, f"fitme_p5_{t}_{ch}.json")

            if hist_name not in root_file:
                print(f"[SKIP] {t.upper()} {ch.upper()}: Histogram '{hist_name}' missing in ROOT file.")
                skipped += 1
                continue
                
            if not os.path.exists(json_file):
                print(f"[SKIP] {t.upper()} {ch.upper()}: JSON file missing (Legacy fit likely failed to converge).")
                skipped += 1
                continue

            with open(json_file, "r") as f:
                d_nom = json.load(f)
                
            fmin, fmax = float(d_nom['fmin']), float(d_nom['fmax'])
            params = d_nom['parameters']
            fit_name = str(d_nom.get('name', 'p5')).upper()

            full_counts = get_atlas_binned_data(root_file[hist_name])
            centers = (ATLAS_BINS[:-1] + ATLAS_BINS[1:]) / 2
            widths = np.diff(ATLAS_BINS)

            mask = (centers >= fmin) & (centers <= fmax)
            c_fit = centers[mask]
            w_fit = widths[mask]
            data_counts = full_counts[mask]

            data_density = data_counts / w_fit
            data_err_density = np.sqrt(np.maximum(data_counts, 1.0)) / w_fit

            fit_legacy_density = ParametricFit(args.sqrts, c_fit, params) / w_fit
            
            min_len = c_fit[0] * 0.05 * 3.0 
            gp_density, ok = fit_gp_density(c_fit, data_density, data_err_density, min_len_scale=min_len)

            if not ok:
                print(f"[SKIP] {t.upper()} {ch.upper()}: GP Fit failed. Data too sparse (Empty high mass tail).")
                skipped += 1
                continue

            pulls_legacy = np.where(data_err_density > 0, (data_density - fit_legacy_density) / data_err_density, 0)
            pulls_gp = np.where(data_err_density > 0, (data_density - gp_density) / data_err_density, 0)

            valid_mask = data_counts > 0
            
            out_base = os.path.join("plots/pull_diagnostics", f"PullDiag_{t}_{ch}")
            title = f"Trigger {t.upper()} | Channel {ch.upper()}"
            
            success, stats_dict, msg = plot_advanced_pull_diagnostics(pulls_legacy[valid_mask], pulls_gp[valid_mask], fit_name, out_base, title)
            
            if success:
                completed += 1
                os.rename(f"{out_base}_Hist.png", os.path.join(out_dir_hist, f"PullDiag_{t}_{ch}_Hist.png"))
                os.rename(f"{out_base}_QQ.png", os.path.join(out_dir_qq, f"PullDiag_{t}_{ch}_QQ.png"))

                results_table.append({
                    'trigger': t.upper(),
                    'channel': ch.upper(),
                    **stats_dict
                })
            else:
                print(f"[SKIP] {t.upper()} {ch.upper()}: {msg}")
                skipped += 1

    # --- TERMINAL OUTPUT BLOCK ---
    print("\n\n" + "="*85)
    print(f"{'PULL DIAGNOSTICS SUMMARY':^85}")
    print("="*85)
    print(f"| {'Trig':<4} | {'Chan':<4} || {'KS p-val (Leg)':<14} | {'KS p-val (GP)':<13} || {'RMS (Leg)':<9} | {'RMS (GP)':<9} |")
    print("-" * 85)
    
    for row in results_table:
        t_str = row['trigger']
        c_str = row['channel']
        ks_leg_str = f"{row['ks_leg']:.3e}" if row['ks_leg'] < 0.01 else f"{row['ks_leg']:.4f}"
        ks_gp_str = f"{row['ks_gp']:.3e}" if row['ks_gp'] < 0.01 else f"{row['ks_gp']:.4f}"
        rms_leg_str = f"{row['std_leg']:.3f}"
        rms_gp_str = f"{row['std_gp']:.3f}"
        print(f"| {t_str:<4} | {c_str:<4} || {ks_leg_str:<14} | {ks_gp_str:<13} || {rms_leg_str:<9} | {rms_gp_str:<9} |")
        
    print("="*85)
    print(f"Diagnostics Finished. Generated: {completed} | Skipped: {skipped}")

if __name__ == "__main__":
    main()
