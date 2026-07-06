#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
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

def get_parametric_fit_func(Ecm):
    """Returns a function suitable for scipy.optimize.curve_fit"""
    def func(x, p0, p1, p2, p3, p4):
        x_scaled = x / Ecm
        nlog = np.log(x_scaled)
        # Add tiny epsilons to prevent invalid exponentiations or log(0)
        safe_base = np.maximum(1.0 - x_scaled, 1e-10)
        safe_x = np.maximum(x_scaled, 1e-10)
        return p0 * np.power(safe_base, p1) * np.power(safe_x, (p2 + p3 * nlog + p4 * nlog * nlog))
    return func

def ParametricFit(Ecm, x_center, p):
    """Wrapper to call the parameteric func natively with a list of parameters"""
    func = get_parametric_fit_func(Ecm)
    return func(x_center, *p[:5])

def get_optimized_zeromean_kernel(centers, density, density_err, min_len_scale_log, max_len_scale_log=5.0):
    """Fits a standard zero-mean GP to the log-density and returns the optimized kernel."""
    X_log = np.log(centers).reshape(-1, 1)
    y_target = np.log(density)
    y_err_target = density_err / density

    kernel = C(1.0, (1e-3, 1e2)) * RBF(
        length_scale=min_len_scale_log, 
        length_scale_bounds=(min_len_scale_log, max_len_scale_log)
    )

    gp = GaussianProcessRegressor(kernel=kernel, alpha=y_err_target**2, n_restarts_optimizer=5, normalize_y=False)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X_log, y_target)
        
    return gp.kernel_

def fit_gp_zeromean_locked(centers, contam_density, contam_err_density, locked_kernel):
    """Fits contaminated data using the fixed kernel parameters from the pure background."""
    X_log = np.log(centers).reshape(-1, 1)
    y_target = np.log(contam_density)
    y_err_target = contam_err_density / contam_density

    # Optimizer=None freezes the hyperparameters
    gp = GaussianProcessRegressor(kernel=locked_kernel, optimizer=None, alpha=y_err_target**2, normalize_y=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X_log, y_target)
        
    y_pred_target = gp.predict(X_log)
    return np.exp(y_pred_target), True

def create_gaussian_signal(centers, widths, mass, res_frac=0.03, significance=5.0, bkg_counts=None):
    """Creates a localized Gaussian signal scaled to the desired significance."""
    sigma = mass * res_frac
    window_mask = (centers >= mass - 2*sigma) & (centers <= mass + 2*sigma)
    bkg_in_window = np.sum(bkg_counts[window_mask]) if bkg_counts is not None else 100
    target_signal_events = significance * np.sqrt(np.maximum(bkg_in_window, 1.0))
    
    pdf = norm.pdf(centers, loc=mass, scale=sigma)
    signal_counts = target_signal_events * (pdf * widths)
    return signal_counts, target_signal_events, sigma

def main():
    parser = argparse.ArgumentParser(description="Compare GP vs 5-Param Signal Injection.")
    parser.add_argument("--fits-dir", type=str, default="/afs/cern.ch/user/e/edweik/private/GlobalSignificanceSearch/fits")
    parser.add_argument("--sqrts", type=float, default=13000.0)
    parser.add_argument("--min-len", type=float, default=0.15)
    parser.add_argument("--max-len", type=float, default=5.0)
    args = parser.parse_args()

    out_dir = "plots/comparison_injection_grid"
    os.makedirs(out_dir, exist_ok=True)

    channels = ["jj", "jb", "bb", "be", "bm", "bg", "je", "jm", "jg"]
    triggers = [f"t{i}" for i in range(1, 8)]

    print(f"\nStarting 9-Panel Comparison Injection Grids")
    print(f"GP Length Scale Bounds: [{args.min_len}, {args.max_len}]")
    print("-" * 60)

    centers = (ATLAS_BINS[:-1] + ATLAS_BINS[1:]) / 2
    widths = np.diff(ATLAS_BINS)
    fit_func_5p = get_parametric_fit_func(args.sqrts)

    completed_grids = 0

    for t in triggers:
        # Initialize Figure for this trigger
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle(f'Signal Absorption Comparison (GP vs 5-Param | 5$\sigma$ Injection)\nTrigger {t.upper()}', fontsize=20, fontweight='bold', y=0.96)
        axes_flat = axes.flatten()

        for i, ch in enumerate(channels):
            ax = axes_flat[i]
            json_file = os.path.join(args.fits_dir, f"fitme_p5_{t}_{ch}.json")

            if not os.path.exists(json_file):
                ax.set_title(f'$M_{{{ch}}}$ (Fit Missing)', fontsize=14)
                ax.axis('off')
                continue

            with open(json_file, "r") as f:
                d_nom = json.load(f)
                
            fmin, fmax = float(d_nom['fmin']), float(d_nom['fmax'])
            params_nom = d_nom['parameters']

            mask = (centers >= fmin) & (centers <= fmax)
            c_fit = centers[mask]
            w_fit = widths[mask]

            # 1. Generate Baseline Asimov Background
            bkg_counts_pure = ParametricFit(args.sqrts, c_fit, params_nom)
            if np.sum(bkg_counts_pure) <= 0:
                ax.set_title(f'$M_{{{ch}}}$ (Zero Background)', fontsize=14)
                ax.axis('off')
                continue

            bkg_density_pure = bkg_counts_pure / w_fit
            bkg_err_density = np.sqrt(np.maximum(bkg_counts_pure, 1.0)) / w_fit

            # 2. Extract the natural kernel from the pure background
            try:
                locked_kernel = get_optimized_zeromean_kernel(
                    c_fit, bkg_density_pure, bkg_err_density, args.min_len, args.max_len
                )
            except Exception:
                ax.set_title(f'$M_{{{ch}}}$ (GP Kernel Opt Failed)', fontsize=14)
                ax.axis('off')
                continue

            mass_span = c_fit[-1] - c_fit[0]
            test_masses = np.linspace(c_fit[0] + 0.15*mass_span, c_fit[-1] - 0.15*mass_span, 10) # 10 test points
            
            absorptions_gp = []
            absorptions_5p = []
            valid_masses = []

            for m_sig in test_masses:
                # Inject 5-sigma signal
                sig_counts, inj_events, sig_width = create_gaussian_signal(
                    c_fit, w_fit, mass=m_sig, res_frac=0.03, significance=5.0, bkg_counts=bkg_counts_pure
                )
                
                if inj_events < 5:
                    continue

                contaminated_counts = bkg_counts_pure + sig_counts
                contam_density = contaminated_counts / w_fit
                contam_err_density = np.sqrt(np.maximum(contaminated_counts, 1.0)) / w_fit

                window = (c_fit >= m_sig - 2*sig_width) & (c_fit <= m_sig + 2*sig_width)
                injected_sum = np.sum(sig_counts[window])

                # --- METHOD 1: FIT GP ---
                try:
                    gp_density, ok_gp = fit_gp_zeromean_locked(c_fit, contam_density, contam_err_density, locked_kernel)
                    if ok_gp:
                        gp_counts = gp_density * w_fit
                        extracted_counts_gp = contaminated_counts - gp_counts
                        extracted_sum_gp = np.sum(extracted_counts_gp[window])
                        abs_gp = (1.0 - (extracted_sum_gp / injected_sum)) * 100.0
                    else:
                        abs_gp = np.nan
                except Exception:
                    abs_gp = np.nan

                # --- METHOD 2: FIT 5-PARAM ---
                try:
                    # Use nominal params as the initial guess to ensure stability
                    popt, _ = curve_fit(
                        fit_func_5p, c_fit, contam_density, 
                        p0=params_nom[:5], sigma=contam_err_density, 
                        absolute_sigma=True, maxfev=10000
                    )
                    density_5p = fit_func_5p(c_fit, *popt)
                    counts_5p = density_5p * w_fit
                    extracted_counts_5p = contaminated_counts - counts_5p
                    extracted_sum_5p = np.sum(extracted_counts_5p[window])
                    abs_5p = (1.0 - (extracted_sum_5p / injected_sum)) * 100.0
                except RuntimeError:
                    # Fit failed to converge
                    abs_5p = np.nan
                except Exception:
                    abs_5p = np.nan

                # Only keep mass point if at least one fit succeeded
                if not np.isnan(abs_gp) or not np.isnan(abs_5p):
                    absorptions_gp.append(abs_gp)
                    absorptions_5p.append(abs_5p)
                    valid_masses.append(m_sig)

            if not valid_masses:
                ax.set_title(f'$M_{{{ch}}}$ (Injection Failed)', fontsize=14)
                ax.axis('off')
                continue

            # ==========================================
            # PLOT: ABSORPTION OVERLAY PANEL
            # ==========================================
            # Matplotlib handles NaNs cleanly (leaves gaps in the line)
            ax.plot(valid_masses, absorptions_gp, 'bo-', lw=2, markersize=6, alpha=0.8, label='GP (Zero-Mean)')
            ax.plot(valid_masses, absorptions_5p, 'rs--', lw=2, markersize=6, alpha=0.8, label='5-Param')
            
            # Threshold lines
            ax.axhline(20.0, color='red', linestyle=':', label='20% Threshold')
            ax.axhline(0.0, color='black', linestyle='-', alpha=0.5)
            
            ax.set_title(f'$M_{{{ch}}}$', fontsize=16, fontweight='bold')
            ax.set_xlabel("Injected Mass $m_{sig}$ [GeV]", fontsize=12)
            ax.set_ylabel("Absorption [%]", fontsize=12)
            
            # Dynamic Y-axis limits
            valid_y = [y for y in absorptions_gp + absorptions_5p if not np.isnan(y)]
            y_limit = max(30.0, max(valid_y) + 5) if valid_y else 30.0
            y_lower = min(-10.0, min(valid_y) - 5) if valid_y else -10.0
            ax.set_ylim(y_lower, y_limit)
            
            ax.legend(fontsize=10, loc='best')
            ax.grid(True, alpha=0.4, linestyle='--')

        # Formatting and Saving Grid
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out_file = os.path.join(out_dir, f"grid_comparison_injection_{t}.png")
        fig.savefig(out_file, bbox_inches='tight', dpi=200)
        plt.close(fig)
        
        completed_grids += 1
        sys.stdout.write(f"\rGenerated {completed_grids}/7 comparison injection grids... ")
        sys.stdout.flush()

    print("\n" + "-" * 60)
    print(f"Finished. Check the {out_dir}/ directory.")

if __name__ == "__main__":
    main()
