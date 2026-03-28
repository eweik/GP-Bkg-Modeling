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

def ParametricFit_alt(Ecm, x_center, p):
    p_safe = np.zeros(5)
    for i in range(min(len(p), 5)):
        p_safe[i] = p[i]
    x = x_center / Ecm
    nlog = np.log(x)
    return p_safe[0] * np.power(np.maximum(1.0 - x, 1e-10), p_safe[1]) * np.power(x, (p_safe[2] + p_safe[3] * nlog + p_safe[4] / np.sqrt(np.maximum(x, 1e-10))))

def format_pval(p):
    """Formats the p-value nicely for the legend."""
    return "<0.001" if p < 0.001 else f"{p:.3f}"

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
    parser = argparse.ArgumentParser(description="Generate 9-panel grids for Advanced GP Spectra and Pull Distributions.")
    parser.add_argument("--root-dir", type=str, default="/afs/cern.ch/user/e/edweik/private/new_ad_files")
    parser.add_argument("--fits-dir", type=str, default="/afs/cern.ch/user/e/edweik/private/GlobalSignificanceSearch/fits")
    parser.add_argument("--sqrts", type=float, default=13000.0)
    parser.add_argument("--min-len", type=float, default=0.15)
    args = parser.parse_args()

    out_dir = "plots/advanced_comparisons_grid"
    os.makedirs(out_dir, exist_ok=True)

    channels = ["jj", "jb", "bb", "be", "bm", "bg", "je", "jm", "jg"]
    triggers = [f"t{i}" for i in range(1, 8)]

    print(f"\nStarting 9-Panel Advanced GP Generation (Length Scale > {args.min_len*100:.1f}%)")
    print("-" * 60)

    for t in triggers:
        root_file_path = os.path.join(args.root_dir, f"data1percent_{t}_HAE_RUN23_nominal_10PB.root")
        root_exists = os.path.exists(root_file_path)
        
        root_file = None
        if root_exists:
            try:
                root_file = uproot.open(root_file_path)
            except Exception as e:
                print(f"[SKIP] {t.upper()}: ROOT read error ({e}).")
                root_file = None

        # Initialize Figures for this trigger
        fig_spec, axes_spec = plt.subplots(3, 3, figsize=(18, 15))
        fig_pull, axes_pull = plt.subplots(3, 3, figsize=(18, 15))
        
        fig_spec.suptitle(f'Spectral Fits - Trigger {t.upper()}', fontsize=20, fontweight='bold', y=0.96)
        fig_pull.suptitle(f'Pull Distributions (Fit vs Data) - Trigger {t.upper()}', fontsize=20, fontweight='bold', y=0.96)

        axes_spec_flat = axes_spec.flatten()
        axes_pull_flat = axes_pull.flatten()

        for i, ch in enumerate(channels):
            ax_spec = axes_spec_flat[i]
            ax_pull = axes_pull_flat[i]

            hist_name = f"M{ch}_data1percent"
            json_file_nom = os.path.join(args.fits_dir, f"fitme_p5_{t}_{ch}.json")
            json_file_alt = os.path.join(args.fits_dir, f"fitme_p5alt_{t}_{ch}.json")

            # Check if Data/Fits exist
            missing_data = False
            if root_file is None or hist_name not in root_file or not os.path.exists(json_file_nom):
                missing_data = True

            if missing_data:
                for ax in [ax_spec, ax_pull]:
                    ax.set_title(f'$M_{{{ch}}}$ (Data/Fit Missing)', fontsize=14)
                    ax.axis('off')
                continue

            # Load Data
            with open(json_file_nom, "r") as f:
                d_nom = json.load(f)
            fmin, fmax = float(d_nom['fmin']), float(d_nom['fmax'])
            params_nom = d_nom['parameters']
            fit_name = str(d_nom.get('name', 'p5')).upper()

            alt_exists = False
            if os.path.exists(json_file_alt):
                with open(json_file_alt, "r") as f:
                    d_alt = json.load(f)
                params_alt = d_alt['parameters']
                alt_exists = True

            full_counts = get_atlas_binned_data(root_file[hist_name])
            centers_full = (ATLAS_BINS[:-1] + ATLAS_BINS[1:]) / 2
            widths_full = np.diff(ATLAS_BINS)

            mask = (centers_full >= fmin) & (centers_full <= fmax)
            c_fit = centers_full[mask]
            w_fit = widths_full[mask]
            data_counts = full_counts[mask]

            data_density = data_counts / w_fit
            data_err_density = np.sqrt(np.maximum(data_counts, 1.0)) / w_fit

            # Evaluate Fits
            fit_leg_counts = ParametricFit(args.sqrts, c_fit, params_nom)
            fit_leg_density = fit_leg_counts / w_fit
            
            if alt_exists:
                fit_alt_counts = ParametricFit_alt(args.sqrts, c_fit, params_alt)
                fit_alt_density = fit_alt_counts / w_fit

            gp_density, gp_err, ok = fit_gp_density_advanced(
                c_fit, data_density, data_err_density, fit_leg_density, min_len_scale_log=args.min_len
            )

            if not ok:
                for ax in [ax_spec, ax_pull]:
                    ax.set_title(f'$M_{{{ch}}}$ (GP Inversion Failed)', fontsize=14)
                    ax.axis('off')
                continue

            # Pulls calculation
            pulls_leg = np.where(data_err_density > 0, (data_density - fit_leg_density) / data_err_density, 0)
            pulls_gp = np.where(data_err_density > 0, (data_density - gp_density) / data_err_density, 0)

            # ==========================================
            # PLOT 1: SPECTRA PANEL
            # ==========================================
            ax_spec.errorbar(c_fit, data_density, yerr=data_err_density, fmt='ko', markersize=4, label='1% Data')
            ax_spec.plot(c_fit, fit_leg_density, 'r-', lw=2, label=f'Nominal Fit ({fit_name})')
            
            if alt_exists:
                ax_spec.plot(c_fit, fit_alt_density, 'g--', lw=2, label='Alternative Fit')
                
            ax_spec.plot(c_fit, gp_density, 'b-', lw=2, label='Advanced GP')
            ax_spec.fill_between(c_fit, gp_density - gp_err, gp_density + gp_err, color='b', alpha=0.2)
            
            ax_spec.set_yscale('log')
            ax_spec.set_title(f'$M_{{{ch}}}$', fontsize=16, fontweight='bold')
            ax_spec.set_xlabel('Invariant Mass [GeV]', fontsize=12)
            ax_spec.set_ylabel('Events / GeV', fontsize=12)
            ax_spec.legend(fontsize=10, loc='upper right') 
            ax_spec.grid(True, which="both", ls="--", alpha=0.3)

            # ==========================================
            # PLOT 2: PULL DISTRIBUTION PANEL
            # ==========================================
            v_leg = pulls_leg[data_counts > 0]
            v_gp = pulls_gp[data_counts > 0]
            x_pdf = np.linspace(-4, 4, 100)
            
            if len(v_leg) > 0 and len(v_gp) > 0:
                for v, color, label in zip([v_leg, v_gp], ['red', 'blue'], [fit_name, 'Advanced GP']):
                    mu, std = norm.fit(v)
                    
                    # Compute KS test against standard normal N(0,1)
                    # 'norm' defaults to loc=0, scale=1
                    ks_stat, p_val = kstest(v, 'norm')
                    
                    ax_pull.hist(v, bins=np.linspace(-4, 4, 30), color=color, alpha=0.4, density=True, 
                                 label=f'{label} ($\mu={mu:.2f}, \sigma={std:.2f}$) [KS p={format_pval(p_val)}]')
                    ax_pull.plot(x_pdf, norm.pdf(x_pdf, mu, std), color=color, ls=':')

                ax_pull.plot(x_pdf, norm.pdf(x_pdf, 0, 1), 'k--', lw=2, label='Ideal $\mathcal{N}(0,1)$')
            
            ax_pull.set_title(f'$M_{{{ch}}}$ Pull Distributions', fontsize=16, fontweight='bold')
            ax_pull.set_xlabel('Pull: (Data - Fit) / Err', fontsize=12)
            ax_pull.set_ylabel('Density', fontsize=12)
            ax_pull.legend(fontsize=9, loc='upper left')

        # Formatting and Saving Grids
        for fig, ftype in zip([fig_spec, fig_pull], ["spectra", "pulls"]):
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            out_file = os.path.join(out_dir, f"grid_{ftype}_{t}.png")
            fig.savefig(out_file, bbox_inches='tight', dpi=200)
            plt.close(fig)

        sys.stdout.write(f"\rGenerated 9-panel grids for {t.upper()}... ")
        sys.stdout.flush()

    print("\n" + "-" * 60)
    print("Finished. Check the plots/advanced_comparisons_grid/ directory.")

if __name__ == "__main__":
    main()
