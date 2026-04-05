#!/usr/bin/env python3
import os
import sys
import json
import uproot
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
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
        p_safe[i] = float(p[i])
    x = x_center / Ecm
    nlog = np.log(x)
    return p_safe[0] * np.power(np.maximum(1.0 - x, 1e-10), p_safe[1]) * np.power(x, (p_safe[2] + p_safe[3] * nlog + p_safe[4] * nlog * nlog))

def fit_5p_floated(centers, density, density_err, Ecm, p0):
    def model(x, p1, p2, p3, p4, p5):
        return ParametricFit(Ecm, x, [p1, p2, p3, p4, p5])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, _ = curve_fit(model, centers, density, p0=p0, sigma=density_err, absolute_sigma=True, maxfev=10000)
        return model(centers, *popt), True
    except Exception:
        return density, False

def get_optimized_background_kernel(centers, density, density_err, parametric_density, min_len_scale_log):
    mask = density > 0
    X_log = np.log(centers[mask]).reshape(-1, 1)
    y_target = np.log(density[mask] / np.maximum(parametric_density[mask], 1e-15))
    y_err_target = density_err[mask] / density[mask]

    kernel = C(1.0, (1e-3, 1e2)) * RBF(length_scale=max(min_len_scale_log * 2.0, 0.5), length_scale_bounds=(min_len_scale_log, 5.0)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=y_err_target**2, n_restarts_optimizer=5, normalize_y=False)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X_log, y_target)
    return gp.kernel_

def fit_gp_locked(centers, contam_density, contam_err_density, parametric_density, locked_kernel):
    mask = contam_density > 0
    if np.sum(mask) < 5:
        return parametric_density, False

    X_log = np.log(centers[mask]).reshape(-1, 1)
    y_target = np.log(contam_density[mask] / np.maximum(parametric_density[mask], 1e-15))
    y_err_target = contam_err_density[mask] / contam_density[mask]
    
    gp = GaussianProcessRegressor(kernel=locked_kernel, optimizer=None, alpha=y_err_target**2, normalize_y=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X_log, y_target)
        
    X_full_log = np.log(centers).reshape(-1, 1)
    y_pred_target = gp.predict(X_full_log)
    return parametric_density * np.exp(y_pred_target), True

def create_gaussian_signal(centers, widths, mass, res_frac=0.03, significance=5.0, bkg_counts=None):
    sigma = mass * res_frac
    window_mask = (centers >= mass - 2*sigma) & (centers <= mass + 2*sigma)
    bkg_in_window = np.sum(bkg_counts[window_mask]) if bkg_counts is not None else 100
    target_signal_events = significance * np.sqrt(np.maximum(bkg_in_window, 1.0))
    
    pdf = norm.pdf(centers, loc=mass, scale=sigma)
    signal_counts = target_signal_events * (pdf * widths)
    return signal_counts, target_signal_events, sigma

def main():
    parser = argparse.ArgumentParser(description="Generate 9-panel grids for GP vs 5p Signal Efficiency.")
    parser.add_argument("-t", "--trigger", type=str, default="all")
    parser.add_argument("-M", "--method", type=str, choices=['asimov', 'toys'], default='toys')
    parser.add_argument("--root-dir", type=str, default="/afs/cern.ch/user/e/edweik/private/new_ad_files")
    parser.add_argument("--fits-dir", type=str, default="/afs/cern.ch/user/e/edweik/private/GlobalSignificanceSearch/fits")
    parser.add_argument("--sqrts", type=float, default=13000.0)
    parser.add_argument("--min-len", type=float, default=0.15)
    parser.add_argument("--toys", type=int, default=20)
    parser.add_argument("--spacing", type=float, default=1.0, help="Spacing of test masses in units of signal sigma (default: 1.0)")
    args = parser.parse_args()

    out_dir = "plots/efficiency_comparisons_grid"
    os.makedirs(out_dir, exist_ok=True)

    channels = ["jj", "jb", "bb", "be", "bm", "bg", "je", "jm", "jg"]
    all_triggers = [f"t{i}" for i in range(1, 8)]
    triggers = all_triggers if args.trigger.lower() == 'all' else [args.trigger.lower()]
    num_iterations = 1 if args.method == 'asimov' else args.toys

    print(f"\nStarting 9-Panel Efficiency Grid Sweep ({args.method.upper()})")
    print(f"Spacing: {args.spacing} sigma steps")
    print("-" * 80)

    completed, skipped = 0, 0
    results_table = []

    for t in triggers:
        root_path = os.path.join(args.root_dir, f"data1percent_{t}_HAE_RUN23_nominal_10PB.root")
        root_exists = os.path.exists(root_path)
        
        root_file = None
        if root_exists:
            try:
                root_file = uproot.open(root_path)
            except Exception as e:
                print(f"[SKIP] {t.upper()}: ROOT read error ({e}).")
                root_file = None

        # Initialize 9-panel figure
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        title_suffix = "Asimov" if args.method == 'asimov' else f"{args.toys} Poisson Toys/Mass"
        fig.suptitle(f'Signal Efficiency Comparison: 5p vs GP - Trigger {t.upper()}\n(5-Sigma Injection, {title_suffix}, Spacing: {args.spacing}$\sigma$)', fontsize=20, fontweight='bold', y=0.96)
        axes_flat = axes.flatten()

        for i, ch in enumerate(channels):
            ax = axes_flat[i]
            hist_name = f"M{ch}_data1percent"
            json_file = os.path.join(args.fits_dir, f"fitme_p5_{t}_{ch}.json")

            if root_file is None or hist_name not in root_file or not os.path.exists(json_file):
                ax.set_title(f'$M_{{{ch}}}$ (Data/Fit Missing)', fontsize=14)
                ax.axis('off')
                skipped += 1
                continue

            with open(json_file, "r") as f:
                d_nom = json.load(f)
                
            fmin, fmax = float(d_nom['fmin']), float(d_nom['fmax'])
            params = d_nom['parameters']
            
            p0 = [float(p) for p in params]
            while len(p0) < 5: p0.append(0.0)

            centers = (ATLAS_BINS[:-1] + ATLAS_BINS[1:]) / 2
            widths = np.diff(ATLAS_BINS)
            mask = (centers >= fmin) & (centers <= fmax)
            c_fit = centers[mask]
            w_fit = widths[mask]

            bkg_counts_pure = ParametricFit(args.sqrts, c_fit, p0)
            if np.sum(bkg_counts_pure) <= 0:
                ax.set_title(f'$M_{{{ch}}}$ (Zero Background)', fontsize=14)
                ax.axis('off')
                skipped += 1
                continue

            bkg_density_pure = bkg_counts_pure / w_fit
            bkg_err_density = np.sqrt(np.maximum(bkg_counts_pure, 1.0)) / w_fit

            try:
                locked_kernel = get_optimized_background_kernel(c_fit, bkg_density_pure, bkg_err_density, bkg_density_pure, args.min_len)
            except Exception:
                ax.set_title(f'$M_{{{ch}}}$ (Kernel Opt Failed)', fontsize=14)
                ax.axis('off')
                skipped += 1
                continue

            # --- DYNAMIC MASS SPACING LOGIC ---
            mass_span = c_fit[-1] - c_fit[0]
            start_mass = c_fit[0] + 0.10 * mass_span
            end_mass = c_fit[-1] - 0.10 * mass_span

            test_masses = []
            current_mass = start_mass
            while current_mass <= end_mass:
                test_masses.append(current_mass)
                # Step forward by args.spacing * sigma (where sigma = 3% of current mass)
                current_mass += current_mass * 0.03 * args.spacing
            
            mean_eff_gp, err_eff_gp = [], []
            mean_eff_5p, err_eff_5p = [], []
            valid_masses = []

            for m_sig in test_masses:
                sig_counts, inj_events, sig_width = create_gaussian_signal(
                    c_fit, w_fit, mass=m_sig, res_frac=0.03, significance=5.0, bkg_counts=bkg_counts_pure
                )
                
                if inj_events < 5:
                    continue

                toy_effs_gp = []
                toy_effs_5p = []

                for toy in range(num_iterations):
                    if args.method == 'asimov':
                        contam_counts = bkg_counts_pure + sig_counts
                    else:
                        contam_counts = np.random.poisson(bkg_counts_pure + sig_counts)
                        
                    contam_density = contam_counts / w_fit
                    contam_err_density = np.sqrt(np.maximum(contam_counts, 1.0)) / w_fit

                    gp_density, gp_ok = fit_gp_locked(c_fit, contam_density, contam_err_density, bkg_density_pure, locked_kernel)
                    fit5_density, p5_ok = fit_5p_floated(c_fit, contam_density, contam_err_density, args.sqrts, p0)

                    if gp_ok and p5_ok:
                        window = (c_fit >= m_sig - 2*sig_width) & (c_fit <= m_sig + 2*sig_width)
                        injected_sum = np.sum(sig_counts[window])

                        gp_ext = np.sum((contam_counts - (gp_density * w_fit))[window])
                        p5_ext = np.sum((contam_counts - (fit5_density * w_fit))[window])
                        
                        toy_effs_gp.append(gp_ext / injected_sum)
                        toy_effs_5p.append(p5_ext / injected_sum)

                if toy_effs_gp and toy_effs_5p:
                    mean_eff_gp.append(np.mean(toy_effs_gp))
                    err_eff_gp.append(np.std(toy_effs_gp) if args.method == 'toys' else 0.0)
                    
                    mean_eff_5p.append(np.mean(toy_effs_5p))
                    err_eff_5p.append(np.std(toy_effs_5p) if args.method == 'toys' else 0.0)
                    
                    valid_masses.append(m_sig)

            if not valid_masses:
                ax.set_title(f'$M_{{{ch}}}$ (Injection Failed)', fontsize=14)
                ax.axis('off')
                skipped += 1
                continue

            results_table.append({
                'trigger': t.upper(),
                'channel': ch.upper(),
                'test_points': len(valid_masses),
                'avg_eff_5p': np.mean(mean_eff_5p),
                'avg_eff_gp': np.mean(mean_eff_gp),
                'min_eff_5p': np.min(mean_eff_5p),
                'min_eff_gp': np.min(mean_eff_gp),
                'avg_rms_5p': np.mean(err_eff_5p),
                'avg_rms_gp': np.mean(err_eff_gp)
            })

            # ==========================================
            # PLOT: EFFICIENCY COMPARISON PANEL
            # ==========================================
            valid_masses_tev = np.array(valid_masses) / 1000.0
            
            ax.errorbar(valid_masses_tev, mean_eff_5p, yerr=err_eff_5p, fmt='ro--', lw=2, capsize=4, markersize=4, label='5-Param (Floated)')
            ax.errorbar(valid_masses_tev, mean_eff_gp, yerr=err_eff_gp, fmt='bo-', lw=2, capsize=4, markersize=4, label=f'GP (Locked)')

            ax.axhline(1.0, color='k', linestyle='--', lw=2)
            ax.axhline(0.8, color='k', linestyle=':', lw=2, label='80% Minimum')
            
            ax.set_title(f"$M_{{{ch}}}$ ({len(valid_masses)} points)", fontsize=16, fontweight='bold')
            ax.set_xlabel(f"Injected Mass [TeV]", fontsize=12)
            ax.set_ylabel("$N_{rec} / N_{inj}$", fontsize=12)
            
            ax.set_ylim(0.1, 1.5)
            ax.legend(loc='lower right', fontsize=10, framealpha=0.9)
            ax.grid(True, alpha=0.3, linestyle='--')

        # Formatting and Saving Grid
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        out_file = os.path.join(out_dir, f"grid_Eff_Comp_{args.method.capitalize()}_{t}.png")
        fig.savefig(out_file, bbox_inches='tight', dpi=200)
        plt.close(fig)
        
        completed += 1
        sys.stdout.write(f"\rProcessed {completed} trigger grids... ")
        sys.stdout.flush()

    print("\n\n" + "=" * 115)
    print(f"{'SIGNAL EFFICIENCY TOY METRICS (N_rec / N_inj)':^115}")
    print("=" * 115)
    print(f"| {'Trig':<4} | {'Chan':<4} | {'Pts':<3} || {'Avg Eff (5p / GP)':<20} || {'Worst Eff (5p / GP)':<22} || {'Avg RMS Spread (5p / GP)':<24} |")
    print("-" * 115)
    for row in results_table:
        avg_eff_str = f"{row['avg_eff_5p']*100:.1f}% / {row['avg_eff_gp']*100:.1f}%"
        min_eff_str = f"{row['min_eff_5p']*100:.1f}% / {row['min_eff_gp']*100:.1f}%"
        rms_str = f"{row['avg_rms_5p']:.3f} / {row['avg_rms_gp']:.3f}"
        print(f"| {row['trigger']:<4} | {row['channel']:<4} | {row['test_points']:<3} || {avg_eff_str:<20} || {min_eff_str:<22} || {rms_str:<24} |")
    print("=" * 115)
    print(f"Sweep Finished. Generated Grids: {completed}")

if __name__ == "__main__":
    main()
