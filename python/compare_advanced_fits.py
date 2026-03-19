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
    p_safe = np.zeros(5)
    for i in range(min(len(p), 5)):
        p_safe[i] = p[i]
    x = x_center / Ecm
    nlog = np.log(x)
    return p_safe[0] * np.power(np.maximum(1.0 - x, 1e-10), p_safe[1]) * np.power(x, (p_safe[2] + p_safe[3] * nlog + p_safe[4] * nlog * nlog))

def fit_gp_density_advanced(centers, density, density_err, parametric_density, min_len_scale_log=0.12):
    mask = density > 0
    if np.sum(mask) < 5:
        return density, np.zeros_like(density), False

    X_log = np.log(centers[mask]).reshape(-1, 1)
    
    y_data = density[mask]
    y_param = parametric_density[mask]
    y_target = np.log(y_data / np.maximum(y_param, 1e-15))
    
    y_err_target = density_err[mask] / y_data

    kernel = C(1.0, (1e-3, 1e2)) * RBF(
        length_scale=max(min_len_scale_log * 2.0, 0.5), 
        length_scale_bounds=(min_len_scale_log, 5.0)
    ) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

    gp = GaussianProcessRegressor(kernel=kernel, alpha=y_err_target**2, n_restarts_optimizer=5, normalize_y=False)

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gp.fit(X_log, y_target)
            
        X_full_log = np.log(centers).reshape(-1, 1)
        y_pred_target, y_pred_target_std = gp.predict(X_full_log, return_std=True)
        
        pred_ratio = np.exp(y_pred_target)
        final_density = parametric_density * pred_ratio
        final_err = final_density * y_pred_target_std
        
        return final_density, final_err, True

    except Exception:
        return density, np.zeros_like(density), False

def get_atlas_binned_data(hist):
    counts, edges = hist.to_numpy()
    centers_raw = (edges[:-1] + edges[1:]) / 2
    binned_counts, _ = np.histogram(centers_raw, bins=ATLAS_BINS, weights=counts)
    return binned_counts

def main():
    parser = argparse.ArgumentParser(description="Compare Parametric fits against Advanced GP models.")
    parser.add_argument("--root-dir", type=str, default="/afs/cern.ch/user/e/edweik/private/new_ad_files")
    parser.add_argument("--fits-dir", type=str, default="/afs/cern.ch/user/e/edweik/private/GlobalSignificanceSearch/fits")
    parser.add_argument("--sqrts", type=float, default=13000.0)
    parser.add_argument("--min-len", type=float, default=0.12, help="Log-space length scale bound")
    args = parser.parse_args()

    out_dir = "plots/advanced_comparisons"
    os.makedirs(out_dir, exist_ok=True)

    channels = ["jj", "jb", "bb", "be", "bm", "bg", "je", "jm", "jg"]
    triggers = [f"t{i}" for i in range(1, 8)]

    completed, skipped = 0, 0

    print(f"\nStarting Advanced GP Spectral Comparison (Length Scale > {args.min_len*100:.1f}%)")
    print("-" * 50)

    for t in triggers:
        root_file_path = os.path.join(args.root_dir, f"data1percent_{t}_HAE_RUN23_nominal_10PB.root")
        
        if not os.path.exists(root_file_path):
            print(f"[SKIP] {t.upper()}: ROOT file missing.")
            skipped += len(channels)
            continue
            
        try:
            root_file = uproot.open(root_file_path)
        except Exception as e:
            print(f"[SKIP] {t.upper()}: ROOT read error ({e}).")
            skipped += len(channels)
            continue

        for ch in channels:
            hist_name = f"M{ch}_data1percent"
            json_file = os.path.join(args.fits_dir, f"fitme_p5_{t}_{ch}.json")

            if hist_name not in root_file:
                print(f"[SKIP] {t.upper()} {ch.upper()}: Histogram missing in ROOT.")
                skipped += 1
                continue
            if not os.path.exists(json_file):
                print(f"[SKIP] {t.upper()} {ch.upper()}: JSON file missing.")
                skipped += 1
                continue

            with open(json_file, "r") as f:
                d_nom = json.load(f)
            
            fmin, fmax = float(d_nom['fmin']), float(d_nom['fmax'])
            params = d_nom['parameters']
            fit_name = str(d_nom.get('name', 'p5')).upper()

            full_counts = get_atlas_binned_data(root_file[hist_name])
            centers_full = (ATLAS_BINS[:-1] + ATLAS_BINS[1:]) / 2
            widths_full = np.diff(ATLAS_BINS)

            mask = (centers_full >= fmin) & (centers_full <= fmax)
            c_fit = centers_full[mask]
            w_fit = widths_full[mask]
            data_counts = full_counts[mask]

            data_density = data_counts / w_fit
            data_err_density = np.sqrt(np.maximum(data_counts, 1.0)) / w_fit

            # Evaluate Legacy Model
            fit_leg_counts = ParametricFit(args.sqrts, c_fit, params)
            fit_leg_density = fit_leg_counts / w_fit
            
            # Evaluate Advanced GP Model
            gp_density, gp_err, ok = fit_gp_density_advanced(
                c_fit, data_density, data_err_density, fit_leg_density, min_len_scale_log=args.min_len
            )

            if not ok:
                print(f"[SKIP] {t.upper()} {ch.upper()}: Advanced GP Matrix Inversion Failed.")
                skipped += 1
                continue

            pulls_leg = np.where(data_err_density > 0, (data_density - fit_leg_density) / data_err_density, 0)
            pulls_gp = np.where(data_err_density > 0, (data_density - gp_density) / data_err_density, 0)

            fig, axes = plt.subplots(4, 1, figsize=(12, 14), gridspec_kw={'height_ratios': [4, 1, 1, 1.5]})
            plt.subplots_adjust(hspace=0.3)

            # Panel 1: Spectra
            ax = axes[0]
            ax.errorbar(c_fit, data_density, yerr=data_err_density, fmt='ko', markersize=4, label='1% Data')
            ax.plot(c_fit, fit_leg_density, 'r-', lw=2, label=f'Legacy Fit ({fit_name})')
            ax.plot(c_fit, gp_density, 'b-', lw=2, label='Advanced GP')
            ax.fill_between(c_fit, gp_density - gp_err, gp_density + gp_err, color='b', alpha=0.2, label='GP Uncertainty')
            ax.set_yscale('log')
            ax.set_ylabel('Events / GeV', fontsize=12)
            ax.set_title(f'Advanced Background Model Comparison: Trigger {t.upper()} | Channel {ch.upper()}', fontsize=14)
            ax.legend(fontsize=12)
            ax.grid(True, which="both", ls="--", alpha=0.3)

            # Panels 2 & 3: Pulls
            for i, (p, color, name) in enumerate(zip([pulls_leg, pulls_gp], ['red', 'blue'], [fit_name, 'Advanced GP'])):
                axes[i+1].axhline(0, color='k', ls='--')
                axes[i+1].scatter(c_fit, p, color=color, s=15)
                axes[i+1].set_ylabel(f'{name} Pulls', fontsize=12)
                axes[i+1].set_ylim(-4, 4)
                axes[i+1].set_xlim(c_fit[0], c_fit[-1])
                
            axes[2].fill_between(c_fit, -1, 1, color='b', alpha=0.1)
            axes[2].set_xlabel('Invariant Mass [GeV]', fontsize=12)

            # Panel 4: Pull Distributions + Gaussian Fits
            ax = axes[3]
            v_leg = pulls_leg[data_counts > 0]
            v_gp = pulls_gp[data_counts > 0]
            x_pdf = np.linspace(-4, 4, 100)
            
            if len(v_leg) > 0 and len(v_gp) > 0:
                for v, color, label in zip([v_leg, v_gp], ['red', 'blue'], [fit_name, 'Advanced GP']):
                    mu, std = norm.fit(v)
                    ax.hist(v, bins=np.linspace(-4, 4, 30), color=color, alpha=0.4, density=True, 
                            label=f'{label} ($\mu={mu:.2f}, \sigma={std:.2f}$)')
                    ax.plot(x_pdf, norm.pdf(x_pdf, mu, std), color=color, ls=':')

                ax.plot(x_pdf, norm.pdf(x_pdf, 0, 1), 'k--', lw=2, label='Ideal $\mathcal{N}(0,1)$')
            
            ax.set_xlabel('Pull: (Data - Fit) / Err', fontsize=12)
            ax.set_ylabel('Density', fontsize=12)
            ax.legend(fontsize=11)

            out_file = os.path.join(out_dir, f"AdvComparison_{t}_{ch}.png")
            plt.savefig(out_file, bbox_inches='tight', dpi=300)
            plt.close()
            
            completed += 1
            sys.stdout.write(f"\rGenerated {completed} plots... ")
            sys.stdout.flush()

    print("\n" + "-" * 50)
    print(f"Finished. Successfully generated: {completed} plots. Skipped: {skipped}")
    print("Check the plots/advanced_comparisons/ directory.")

if __name__ == "__main__":
    main()
