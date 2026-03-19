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

def get_optimized_background_kernel(centers, density, density_err, parametric_density, min_len_scale_log):
    """STEP 1: Find the natural length scale of the pure background."""
    X_log = np.log(centers).reshape(-1, 1)
    y_target = np.log(density / np.maximum(parametric_density, 1e-15))
    y_err_target = density_err / density

    kernel = C(1.0, (1e-3, 1e2)) * RBF(
        length_scale=max(min_len_scale_log * 2.0, 0.5), 
        length_scale_bounds=(min_len_scale_log, 5.0)
    ) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))

    gp = GaussianProcessRegressor(kernel=kernel, alpha=y_err_target**2, n_restarts_optimizer=5, normalize_y=False)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X_log, y_target)
        
    return gp.kernel_

def fit_gp_locked(centers, contam_density, contam_err_density, parametric_density, locked_kernel):
    """STEP 2: Fit the contaminated data using the LOCKED background kernel."""
    X_log = np.log(centers).reshape(-1, 1)
    y_target = np.log(contam_density / np.maximum(parametric_density, 1e-15))
    y_err_target = contam_err_density / contam_density

    # Optimizer=None freezes the hyperparameters exactly where they are
    gp = GaussianProcessRegressor(kernel=locked_kernel, optimizer=None, alpha=y_err_target**2, normalize_y=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X_log, y_target)
        
    y_pred_target = gp.predict(X_log)
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
    parser = argparse.ArgumentParser(description="Advanced GP Signal Injection Test (Locked Kernel).")
    parser.add_argument("--fits-dir", type=str, required=True)
    parser.add_argument("--sqrts", type=float, default=13000.0)
    parser.add_argument("--min-len", type=float, default=0.15, help="Log-space length scale bound")
    args = parser.parse_args()

    out_dir = "plots/injection_tests"
    os.makedirs(out_dir, exist_ok=True)

    channels = ["jj", "jb", "bb", "be", "bm", "bg", "je", "jm", "jg"]
    triggers = [f"t{i}" for i in range(1, 8)]

    print(f"\nStarting LOCKED Kernel Signal Injection Sweep (Length Scale Bound: {args.min_len*100:.1f}%)")
    print("-" * 60)

    centers = (ATLAS_BINS[:-1] + ATLAS_BINS[1:]) / 2
    widths = np.diff(ATLAS_BINS)

    completed = 0
    results_table = []

    for t in triggers:
        for ch in channels:
            json_file = os.path.join(args.fits_dir, f"fitme_p5_{t}_{ch}.json")
            if not os.path.exists(json_file):
                continue

            with open(json_file, "r") as f:
                d_nom = json.load(f)
                
            fmin, fmax = float(d_nom['fmin']), float(d_nom['fmax'])
            params = d_nom['parameters']

            mask = (centers >= fmin) & (centers <= fmax)
            c_fit = centers[mask]
            w_fit = widths[mask]

            # 1. Generate Baseline Asimov Background
            bkg_counts_pure = ParametricFit(args.sqrts, c_fit, params)
            if np.sum(bkg_counts_pure) <= 0:
                continue

            bkg_density_pure = bkg_counts_pure / w_fit
            bkg_err_density = np.sqrt(np.maximum(bkg_counts_pure, 1.0)) / w_fit

            # 2. Extract the natural kernel from the pure background
            try:
                locked_kernel = get_optimized_background_kernel(
                    c_fit, bkg_density_pure, bkg_err_density, bkg_density_pure, args.min_len
                )
            except Exception:
                continue # Skip if base optimization fails

            mass_span = c_fit[-1] - c_fit[0]
            test_masses = np.linspace(c_fit[0] + 0.15*mass_span, c_fit[-1] - 0.15*mass_span, 5)
            
            absorptions = []
            valid_masses = []

            for m_sig in test_masses:
                sig_counts, inj_events, sig_width = create_gaussian_signal(
                    c_fit, w_fit, mass=m_sig, res_frac=0.03, significance=5.0, bkg_counts=bkg_counts_pure
                )
                
                if inj_events < 5:
                    continue

                contaminated_counts = bkg_counts_pure + sig_counts
                contam_density = contaminated_counts / w_fit
                contam_err_density = np.sqrt(np.maximum(contaminated_counts, 1.0)) / w_fit

                # 3. Fit contaminated data with LOCKED kernel
                try:
                    gp_density, ok = fit_gp_locked(c_fit, contam_density, contam_err_density, bkg_density_pure, locked_kernel)
                except Exception:
                    continue

                if not ok:
                    continue

                gp_counts = gp_density * w_fit
                extracted_counts = contaminated_counts - gp_counts
                
                window = (c_fit >= m_sig - 2*sig_width) & (c_fit <= m_sig + 2*sig_width)
                extracted_sum = np.sum(extracted_counts[window])
                injected_sum = np.sum(sig_counts[window])
                
                absorption = 1.0 - (extracted_sum / injected_sum)
                
                absorptions.append(absorption * 100.0)
                valid_masses.append(m_sig)

            if not valid_masses:
                continue

            # Record stats
            max_abs = max(absorptions)
            max_abs_mass = valid_masses[absorptions.index(max_abs)]
            
            if max_abs <= 20.0:
                status = "PASS"
            elif max_abs <= 30.0:
                status = "WARN"
            else:
                status = "FAIL"

            results_table.append({
                'trigger': t.upper(),
                'channel': ch.upper(),
                'max_abs': max_abs,
                'max_mass': max_abs_mass,
                'status': status
            })

            # --- PLOTTING ---
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(valid_masses, absorptions, 'bo-', lw=2, markersize=8, label='Signal Absorption')
            
            ax.axhline(20.0, color='r', linestyle='--', label='ATLAS 20% Threshold')
            ax.axhline(0.0, color='k', linestyle='-')
            
            ax.set_title(f"LOCKED GP Signal Absorption: {t.upper()} {ch.upper()}\n(5-Sigma Injection, 3% Mass Resolution)", fontsize=14)
            ax.set_xlabel("Injected Resonance Mass $m_{sig}$ [GeV]", fontsize=12)
            ax.set_ylabel("Absorption Factor [%]", fontsize=12)
            
            y_limit = max(30.0, max(absorptions) + 5)
            ax.set_ylim(-10, y_limit)
            
            ax.legend(fontsize=11)
            ax.grid(True, alpha=0.4, linestyle='--')
            
            out_file = os.path.join(out_dir, f"Injection_{t}_{ch}.png")
            plt.savefig(out_file, bbox_inches='tight', dpi=300)
            plt.close()
            completed += 1
            
            sys.stdout.write(f"\rProcessed {completed} channel curves... ")
            sys.stdout.flush()

    # --- TERMINAL OUTPUT BLOCK ---
    print("\n\n" + "="*65)
    print(f"{'LOCKED KERNEL INJECTION RESULTS':^65}")
    print("="*65)
    print(f"| {'Trig':<4} | {'Chan':<4} || {'Max Absorption':<14} | {'@ Mass (GeV)':<12} | {'Status':<6} |")
    print("-" * 65)
    
    for row in results_table:
        t_str = row['trigger']
        c_str = row['channel']
        abs_str = f"{row['max_abs']:.1f}%"
        mass_str = f"{row['max_mass']:.0f}"
        status = row['status']
        
        print(f"| {t_str:<4} | {c_str:<4} || {abs_str:<14} | {mass_str:<12} | {status:<6} |")
        
    print("="*65)
    print(f"Finished. Generated {completed} absorption curves.")

if __name__ == "__main__":
    main()
