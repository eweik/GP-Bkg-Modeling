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

def ParametricFit(Ecm, x_center, p):
    x = x_center / Ecm
    nlog = np.log(x)
    return p[0] * np.power(np.maximum(1.0 - x, 1e-10), p[1]) * np.power(x, (p[2] + p[3] * nlog + p[4] * nlog * nlog))

def get_fixed_p5_func(Ecm, fixed_p3, fixed_p4):
    """Returns a 5-param function where p3 and p4 are locked (Constants)."""
    def func(x, p0, p1, p2):
        x_scaled = x / Ecm
        nlog = np.log(x_scaled)
        safe_base = np.maximum(1.0 - x_scaled, 1e-10)
        safe_x = np.maximum(x_scaled, 1e-10)
        return p0 * np.power(safe_base, p1) * np.power(safe_x, (p2 + fixed_p3 * nlog + fixed_p4 * nlog * nlog))
    return func

def get_optimized_zeromean_kernel(centers, density, density_err, min_len, max_len):
    """Fits standard zero-mean GP to the pure Asimov log-density to get baseline kernel."""
    X_log = np.log(centers).reshape(-1, 1)
    y_target = np.log(density)
    y_err_target = density_err / density
    kernel = C(1.0, (1e-3, 1e2)) * RBF(length_scale=min_len, length_scale_bounds=(min_len, max_len))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=y_err_target**2, n_restarts_optimizer=5, normalize_y=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X_log, y_target)
    return gp.kernel_

def fit_gp_zeromean_locked(centers, contam_density, contam_err_density, locked_kernel):
    """Fits toy data using the fixed kernel parameters to prevent overfitting noise."""
    X_log = np.log(centers).reshape(-1, 1)
    y_target = np.log(contam_density)
    y_err_target = contam_err_density / contam_density
    gp = GaussianProcessRegressor(kernel=locked_kernel, optimizer=None, alpha=y_err_target**2, normalize_y=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X_log, y_target)
    return np.exp(gp.predict(X_log)), True

def create_gaussian_signal(centers, widths, mass, res_frac, significance=5.0, bkg_counts=None):
    sigma = mass * res_frac
    window_mask = (centers >= mass - 2*sigma) & (centers <= mass + 2*sigma)
    bkg_in_window = np.sum(bkg_counts[window_mask]) if bkg_counts is not None else 100
    target_signal_events = significance * np.sqrt(np.maximum(bkg_in_window, 1.0))
    pdf = norm.pdf(centers, loc=mass, scale=sigma)
    return target_signal_events * (pdf * widths), target_signal_events, sigma

def main():
    parser = argparse.ArgumentParser(description="Toy MC Signal Injection: GP vs Fixed-Param.")
    parser.add_argument("--fits-dir", type=str, default="fits")
    parser.add_argument("--sqrts", type=float, default=13000.0)
    parser.add_argument("--n-toys", type=int, default=100)
    parser.add_argument("--min-len", type=float, default=0.15)
    parser.add_argument("--max-len", type=float, default=5.0)
    args = parser.parse_args()

    out_dir = "plots/toy_injection_gp_vs_param"
    os.makedirs(out_dir, exist_ok=True)

    channels = ["jj", "jb", "bb", "be", "bm", "bg", "je", "jm", "jg"]
    triggers = [f"t{i}" for i in range(1, 8)]
    widths_to_test = [0.05, 0.15] 

    centers = (ATLAS_BINS[:-1] + ATLAS_BINS[1:]) / 2
    widths = np.diff(ATLAS_BINS)

    print(f"\nStarting Toy MC Injection: GP vs Parametric (N_toys = {args.n_toys})")
    print("-" * 70)

    for t in triggers:
        fig, axes = plt.subplots(3, 3, figsize=(20, 16))
        fig.suptitle(f'Signal Recovery ($5\sigma$ Toys) - GP vs 3-Param | Trigger {t.upper()} | $N_{{toys}}={args.n_toys}$', fontsize=22, fontweight='bold', y=0.96)
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
            params_nom = [float(p) for p in d_nom['parameters']]

            mask = (centers >= fmin) & (centers <= fmax)
            c_fit, w_fit = centers[mask], widths[mask]

            bkg_counts_pure = ParametricFit(args.sqrts, c_fit, params_nom)
            if np.sum(bkg_counts_pure) <= 0:
                ax.set_title(f'$M_{{{ch}}}$ (Zero Background)', fontsize=14)
                ax.axis('off')
                continue

            # Extract baseline models
            bkg_density_pure = bkg_counts_pure / w_fit
            bkg_err_density = np.sqrt(np.maximum(bkg_counts_pure, 1.0)) / w_fit
            
            try:
                locked_kernel = get_optimized_zeromean_kernel(c_fit, bkg_density_pure, bkg_err_density, args.min_len, args.max_len)
            except Exception:
                ax.set_title(f'$M_{{{ch}}}$ (GP Init Failed)', fontsize=14)
                ax.axis('off')
                continue

            fit_func_fixed = get_fixed_p5_func(args.sqrts, params_nom[3], params_nom[4])
            mass_span = c_fit[-1] - c_fit[0]
            test_masses = np.linspace(c_fit[0] + 0.15*mass_span, c_fit[-1] - 0.15*mass_span, 8)
            
            results = {w: {'masses': [], 'param_mean': [], 'param_std': [], 'gp_mean': [], 'gp_std': []} for w in widths_to_test}

            for res_frac in widths_to_test:
                for m_sig in test_masses:
                    sys.stdout.write(f"\rProcessing {t.upper()}_{ch.upper()} | Width: {res_frac*100:2.0f}% | Mass: {m_sig:4.0f} GeV... ")
                    sys.stdout.flush()

                    sig_counts, inj_events, sig_width = create_gaussian_signal(c_fit, w_fit, m_sig, res_frac, 5.0, bkg_counts_pure)
                    if inj_events < 5: continue
                        
                    expected_counts = bkg_counts_pure + sig_counts
                    window = (c_fit >= m_sig - 2*sig_width) & (c_fit <= m_sig + 2*sig_width)
                    injected_sum = np.sum(sig_counts[window])

                    effs_param, effs_gp = [], []
                    
                    for toy in range(args.n_toys):
                        toy_counts = np.random.poisson(expected_counts)
                        toy_density = toy_counts / w_fit
                        toy_err_density = np.sqrt(np.maximum(toy_counts, 1.0)) / w_fit

                        # 1. Fit Parametric
                        try:
                            with warnings.catch_warnings():
                                warnings.simplefilter("ignore")
                                popt, _ = curve_fit(fit_func_fixed, c_fit, toy_density, p0=params_nom[:3], sigma=toy_err_density, absolute_sigma=True, maxfev=5000)
                            fit_counts = fit_func_fixed(c_fit, *popt) * w_fit
                            effs_param.append((np.sum((toy_counts - fit_counts)[window]) / injected_sum) * 100.0)
                        except RuntimeError: pass

                        # 2. Fit GP
                        try:
                            gp_density, _ = fit_gp_zeromean_locked(c_fit, toy_density, toy_err_density, locked_kernel)
                            gp_counts = gp_density * w_fit
                            effs_gp.append((np.sum((toy_counts - gp_counts)[window]) / injected_sum) * 100.0)
                        except Exception: pass
                    
                    if effs_param and effs_gp:
                        results[res_frac]['masses'].append(m_sig)
                        results[res_frac]['param_mean'].append(np.mean(effs_param))
                        results[res_frac]['param_std'].append(np.std(effs_param))
                        results[res_frac]['gp_mean'].append(np.mean(effs_gp))
                        results[res_frac]['gp_std'].append(np.std(effs_gp))

            # Plotting Configuration
            colors = {0.05: ('blue', 'cyan'), 0.15: ('red', 'orange')} # (Param color, GP color)
            
            for res_frac in widths_to_test:
                x = results[res_frac]['masses']
                if not x: continue
                
                # Plot Parametric
                ax.errorbar(x, results[res_frac]['param_mean'], yerr=results[res_frac]['param_std'], 
                            fmt=f"{colors[res_frac][0]}s--", lw=2, capsize=3, alpha=0.8, 
                            label=f'Param ({int(res_frac*100)}% W)')
                # Plot GP
                ax.errorbar(x, results[res_frac]['gp_mean'], yerr=results[res_frac]['gp_std'], 
                            fmt=f"{colors[res_frac][1]}o-", lw=2, capsize=3, alpha=0.8, 
                            label=f'GP ({int(res_frac*100)}% W)')

            ax.axhline(100.0, color='green', linestyle='-', lw=1.5, alpha=0.8)
            ax.axhline(80.0, color='black', linestyle='--', alpha=0.5)

            ax.set_title(f'$M_{{{ch}}}$', fontsize=16, fontweight='bold')
            ax.set_xlabel("Injected Mass $m_{sig}$ [GeV]", fontsize=12)
            ax.set_ylabel("$N_{rec} / N_{inj}$ [%]", fontsize=12)
            ax.set_ylim(20, 180)
            ax.grid(True, linestyle='--', alpha=0.4)
            if i == 0: ax.legend(fontsize=10, loc='best', ncol=2)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out_file = os.path.join(out_dir, f"grid_toy_gp_vs_param_{t}.png")
        fig.savefig(out_file, bbox_inches='tight', dpi=200)
        plt.close(fig)

    print("\n" + "-" * 70)
    print(f"Finished. Check the {out_dir}/ directory.")

if __name__ == "__main__":
    main()
