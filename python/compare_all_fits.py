#!/usr/bin/env python3
import os
import sys
import json
import uproot
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
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
    """Safely evaluates the ATLAS background model up to 5 parameters."""
    # Ensure parameter array is exactly length 5, padding with 0.0 if necessary
    p_safe = np.zeros(5)
    for i in range(min(len(p), 5)):
        p_safe[i] = p[i]
        
    x = x_center / Ecm
    nlog = np.log(x)
    return p_safe[0] * np.power(np.maximum(1.0 - x, 1e-10), p_safe[1]) * np.power(x, (p_safe[2] + p_safe[3] * nlog + p_safe[4] * nlog * nlog))

def fit_gp_density(centers, density, density_err, min_len_scale=300.0):
    """Fits GP to density natively."""
    mask = density > 0
    if np.sum(mask) < 5:
        return density, np.zeros_like(density), False

    X = centers[mask].reshape(-1, 1)
    y = density[mask]
    y_err = density_err[mask]

    y_log = np.log(y)
    y_err_log = y_err / y

    kernel = C(1.0, (1e-3, 1e3)) * RBF(
        length_scale=max(min_len_scale * 2, 1000.0), 
        length_scale_bounds=(min_len_scale, 1e4)
    ) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

    gp = GaussianProcessRegressor(kernel=kernel, alpha=y_err_log**2, n_restarts_optimizer=5, normalize_y=True)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(X, y_log)
        
        y_pred_log, y_pred_log_std = gp.predict(centers.reshape(-1, 1), return_std=True)
        return np.exp(y_pred_log), np.exp(y_pred_log) * y_pred_log_std, True
    except Exception:
        return density, np.zeros_like(density), False

def get_atlas_binned_data(hist):
    counts, edges = hist.to_numpy()
    centers_raw = (edges[:-1] + edges[1:]) / 2
    binned_counts, _ = np.histogram(centers_raw, bins=ATLAS_BINS, weights=counts)
    return binned_counts

def main():
    parser = argparse.ArgumentParser(description="Compare Parametric fits against GP models.")
    parser.add_argument("--root-dir", type=str, 
                        default="/afs/cern.ch/user/e/edweik/private/new_ad_files")
    parser.add_argument("--fits-dir", type=str, 
                        default="/afs/cern.ch/user/e/edweik/private/GlobalSignificanceSearch/fits")
    parser.add_argument("--sqrts", type=float, default=13000.0,
                        help="Center of mass energy used in legacy function.")
    args = parser.parse_args()

    root_dir = args.root_dir
    fits_dir = args.fits_dir
    sqrts = args.sqrts
    out_dir = "plots/comparisons"
    os.makedirs(out_dir, exist_ok=True)

    channels = ["jj", "jb", "bb", "je", "jm", "be", "bm", "jg", "bg"]
    triggers = [f"t{i}" for i in range(1, 8)]

    total_plots = len(triggers) * len(channels)
    completed = 0
    skipped_count = 0

    print("\nStarting comparison sweep...")
    print("-" * 50)

    for t in triggers:
        root_file_path = os.path.join(root_dir, f"data1percent_{t}_HAE_RUN23_nominal_10PB.root")
        
        if not os.path.exists(root_file_path):
            print(f"[SKIP] {t}: ROOT file missing.")
            skipped_count += 9
            continue
            
        try:
            root_file = uproot.open(root_file_path)
        except Exception as e:
            print(f"[SKIP] {t}: Could not open ROOT file ({e}).")
            skipped_count += 9
            continue

        for ch in channels:
            hist_name = f"M{ch}_data1percent"
            json_file = os.path.join(fits_dir, f"fitme_p5_{t}_{ch}.json")

            if hist_name not in root_file:
                print(f"[SKIP] {t} {ch}: Histogram '{hist_name}' missing in ROOT file.")
                skipped_count += 1
                continue
                
            if not os.path.exists(json_file):
                print(f"[SKIP] {t} {ch}: JSON file missing (Legacy fit likely failed).")
                skipped_count += 1
                continue

            with open(json_file, "r") as f:
                d_nom = json.load(f)
                
            fmin_val = float(d_nom['fmin'])
            fmax_val = float(d_nom['fmax'])
            params = d_nom['parameters']
            # Dynamically grab the fit name to display in the legend
            fit_name = str(d_nom.get('name', 'p5')).upper()

            full_counts = get_atlas_binned_data(root_file[hist_name])
            centers_full = (ATLAS_BINS[:-1] + ATLAS_BINS[1:]) / 2
            widths_full = np.diff(ATLAS_BINS)

            mask = (centers_full >= fmin_val) & (centers_full <= fmax_val)
            c_fit = centers_full[mask]
            w_fit = widths_full[mask]
            data_counts = full_counts[mask]

            # Calculate Density (Events / GeV) for principled GP fitting
            data_density = data_counts / w_fit
            data_err_density = np.sqrt(np.maximum(data_counts, 1.0)) / w_fit

            # Evaluate Legacy Model
            fit_legacy_counts = ParametricFit(sqrts, c_fit, params)
            fit_legacy_density = fit_legacy_counts / w_fit
            
            # Run the GP Fit natively on density
            min_len = c_fit[0] * 0.05 * 3.0 
            gp_density, gp_err, ok = fit_gp_density(c_fit, data_density, data_err_density, min_len_scale=min_len)

            if not ok:
                print(f"[SKIP] {t} {ch}: GP failed. Data too sparse (Empty high mass tail).")
                skipped_count += 1
                continue

            pulls_legacy = np.where(data_err_density > 0, (data_density - fit_legacy_density) / data_err_density, 0)
            pulls_gp = np.where(data_err_density > 0, (data_density - gp_density) / data_err_density, 0)

            fig, axes = plt.subplots(4, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [4, 1, 1, 1.5]})
            plt.subplots_adjust(hspace=0.3)

            ax = axes[0]
            ax.errorbar(c_fit, data_density, yerr=data_err_density, fmt='ko', markersize=4, label='1% Data')
            ax.plot(c_fit, fit_legacy_density, 'r-', lw=2, label=f'Legacy Fit ({fit_name})')
            ax.plot(c_fit, gp_density, 'b-', lw=2, label='Gaussian Process')
            ax.fill_between(c_fit, gp_density - gp_err, gp_density + gp_err, color='b', alpha=0.2, label='GP Uncertainty')
            ax.set_yscale('log')
            ax.set_ylabel('Events / GeV', fontsize=12)
            ax.set_title(f'Background Model Comparison: Trigger {t.upper()} | Channel {ch.upper()}', fontsize=14)
            ax.legend(fontsize=12)
            ax.grid(True, which="both", ls="--", alpha=0.3)

            ax = axes[1]
            ax.axhline(0, color='k', ls='--')
            ax.scatter(c_fit, pulls_legacy, color='r', s=15)
            ax.set_ylabel(f'{fit_name} Pulls', fontsize=12)
            ax.set_ylim(-4, 4)
            ax.set_xlim(c_fit[0], c_fit[-1])

            ax = axes[2]
            ax.axhline(0, color='k', ls='--')
            ax.scatter(c_fit, pulls_gp, color='b', s=15)
            ax.fill_between(c_fit, -1, 1, color='b', alpha=0.1)
            ax.set_ylabel('GP Pulls', fontsize=12)
            ax.set_xlabel('Invariant Mass [GeV]', fontsize=12)
            ax.set_ylim(-4, 4)
            ax.set_xlim(c_fit[0], c_fit[-1])

            ax = axes[3]
            bins = np.linspace(-4, 4, 30)
            
            valid_pulls_leg = pulls_legacy[data_counts > 0]
            valid_pulls_gp = pulls_gp[data_counts > 0]
            
            if len(valid_pulls_leg) > 0 and len(valid_pulls_gp) > 0:
                mu_leg, std_leg = norm.fit(valid_pulls_leg)
                mu_gp, std_gp = norm.fit(valid_pulls_gp)
                
                ax.hist(valid_pulls_leg, bins=bins, color='r', alpha=0.4, density=True, label=f'{fit_name} ($\mu$={mu_leg:.2f}, $\sigma$={std_leg:.2f})')
                ax.hist(valid_pulls_gp, bins=bins, color='b', alpha=0.4, density=True, label=f'GP ($\mu$={mu_gp:.2f}, $\sigma$={std_gp:.2f})')
                
                x_pdf = np.linspace(-4, 4, 100)
                ax.plot(x_pdf, norm.pdf(x_pdf, 0, 1), 'k--', lw=2, label='Standard Normal $\mathcal{N}(0,1)$')
            
            ax.set_xlabel('Pull: (Data - Fit) / Err', fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.legend(fontsize=11)

            out_file = os.path.join(out_dir, f"Comparison_{t}_{ch}.png")
            plt.savefig(out_file, bbox_inches='tight', dpi=300)
            plt.close()
            
            completed += 1

    print("-" * 50)
    print(f"Sweep Finished.")
    print(f"Successfully generated: {completed} plots.")
    print(f"Skipped combinations:   {skipped_count}")
    print("Check the plots/comparisons/ directory for results.")

if __name__ == "__main__":
    main()
