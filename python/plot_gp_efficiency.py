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
    X_log = np.log(centers).reshape(-1, 1)
    y_target = np.log(contam_density / np.maximum(parametric_density, 1e-15))
    y_err_target = contam_err_density / contam_density

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
    parser = argparse.ArgumentParser(description="Plot GP Signal Efficiency.")
    parser.add_argument("-t", "--trigger", type=str, required=True, help="Trigger (e.g., t1)")
    parser.add_argument("-c", "--channel", type=str, required=True, help="Channel (e.g., jj)")
    parser.add_argument("--root-dir", type=str, default="/afs/cern.ch/user/e/edweik/private/new_ad_files")
    parser.add_argument("--fits-dir", type=str, default="/afs/cern.ch/user/e/edweik/private/GlobalSignificanceSearch/fits")
    parser.add_argument("--sqrts", type=float, default=13000.0)
    args = parser.parse_args()

    t = args.trigger.lower()
    ch = args.channel.lower()
    out_dir = "plots/efficiency_plots"
    os.makedirs(out_dir, exist_ok=True)

    print(f"\nEvaluating Advanced GP Efficiency for {t.upper()} {ch.upper()}")
    print("-" * 55)

    root_path = os.path.join(args.root_dir, f"data1percent_{t}_HAE_RUN23_nominal_10PB.root")
    json_file = os.path.join(args.fits_dir, f"fitme_p5_{t}_{ch}.json")

    if not os.path.exists(root_path) or not os.path.exists(json_file):
        print("ERROR: Missing ROOT or JSON file for this combination.")
        sys.exit(1)

    with open(json_file, "r") as f:
        d_nom = json.load(f)
        
    fmin, fmax = float(d_nom['fmin']), float(d_nom['fmax'])
    params = d_nom['parameters']

    centers = (ATLAS_BINS[:-1] + ATLAS_BINS[1:]) / 2
    widths = np.diff(ATLAS_BINS)

    mask = (centers >= fmin) & (centers <= fmax)
    c_fit = centers[mask]
    w_fit = widths[mask]

    bkg_counts_pure = ParametricFit(args.sqrts, c_fit, params)
    bkg_density_pure = bkg_counts_pure / w_fit
    bkg_err_density = np.sqrt(np.maximum(bkg_counts_pure, 1.0)) / w_fit

    # Evaluate two different length scales
    len_scales = [0.12, 0.15]
    efficiencies = {0.12: [], 0.15: []}
    
    mass_span = c_fit[-1] - c_fit[0]
    test_masses = np.linspace(c_fit[0] + 0.10*mass_span, c_fit[-1] - 0.10*mass_span, 10)
    valid_masses = []

    for ls in len_scales:
        print(f"Locking GP Kernel at Min Length Scale = {ls*100:.1f}%...")
        try:
            locked_kernel = get_optimized_background_kernel(c_fit, bkg_density_pure, bkg_err_density, bkg_density_pure, ls)
        except Exception as e:
            print(f"Failed to optimize background kernel for scale {ls}: {e}")
            sys.exit(1)

        for m_sig in test_masses:
            sig_counts, inj_events, sig_width = create_gaussian_signal(
                c_fit, w_fit, mass=m_sig, res_frac=0.03, significance=5.0, bkg_counts=bkg_counts_pure
            )
            
            if inj_events < 5:
                continue
                
            if m_sig not in valid_masses and ls == len_scales[0]:
                valid_masses.append(m_sig)

            contam_counts = bkg_counts_pure + sig_counts
            contam_density = contam_counts / w_fit
            contam_err_density = np.sqrt(np.maximum(contam_counts, 1.0)) / w_fit

            try:
                gp_density, ok = fit_gp_locked(c_fit, contam_density, contam_err_density, bkg_density_pure, locked_kernel)
            except Exception:
                continue

            if not ok:
                continue

            gp_counts = gp_density * w_fit
            extracted_counts = contam_counts - gp_counts
            
            window = (c_fit >= m_sig - 2*sig_width) & (c_fit <= m_sig + 2*sig_width)
            extracted_sum = np.sum(extracted_counts[window])
            injected_sum = np.sum(sig_counts[window])
            
            # Legacy script plots Efficiency = Extracted / Injected
            efficiency = extracted_sum / injected_sum
            efficiencies[ls].append(efficiency)

    if not valid_masses:
        print("Not enough data to perform injection test in this phase space.")
        sys.exit(1)

    # --- PLOTTING ---
    # Convert masses to TeV to match legacy plot
    valid_masses_tev = np.array(valid_masses) / 1000.0

    fig, ax = plt.subplots(figsize=(8, 7))
    
    # 0.12 Length Scale (Red Dashed)
    ax.plot(valid_masses_tev, efficiencies[0.12], 'ro--', lw=2, markersize=8, label='L = 0.12 (12% width)')
    
    # 0.15 Length Scale (Blue Solid)
    ax.plot(valid_masses_tev, efficiencies[0.15], 'bo-', lw=2, markersize=8, label='L = 0.15 (15% width)')

    # Ideal lines
    ax.axhline(1.0, color='k', linestyle='--', lw=2) # 100% Efficiency
    ax.axhline(0.8, color='k', linestyle=':', lw=2, label='ATLAS Minimum (80% Eff)') # 80% Efficiency (20% absorption limit)
    
    # Text Annotations mirroring legacy script
    ax.text(0.65, 0.25, 'Advanced GP', transform=ax.transAxes, fontsize=12, fontweight='bold')
    ax.text(0.65, 0.20, 'Locked Kernel Fit', transform=ax.transAxes, fontsize=12)
    ax.text(0.65, 0.15, 'Asimov Injection', transform=ax.transAxes, fontsize=12)

    ax.set_title(f"Signal Extraction Efficiency\nTrigger: {t.upper()} | Channel: {ch.upper()}", fontsize=14, fontweight='bold')
    ax.set_xlabel(f"m_{{{ch.lower()}}} [TeV]", fontsize=14)
    ax.set_ylabel("$N_{rec} / N_{inj}$", fontsize=14)
    
    # Enforce legacy Y-axis limits
    ax.set_ylim(0.1, 1.4)
    
    # Styling tweaks
    ax.legend(loc='lower left', fontsize=12, framealpha=0.9)
    ax.tick_params(axis='both', which='major', labelsize=12, direction='in', length=6)
    ax.grid(True, alpha=0.3, linestyle='--')

    out_file = os.path.join(out_dir, f"Efficiency_{t}_{ch}.png")
    plt.savefig(out_file, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"Efficiency plot saved to {out_file}")
    print("======================================================\n")

if __name__ == "__main__":
    main()
