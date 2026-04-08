#!/usr/bin/env python3
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Calculate Trigger-Wide Global Significance.")
    parser.add_argument("--trigger", type=str, default="t2",
                        help="Target trigger to evaluate (default: t2)")
    parser.add_argument("--ExpectedLocalZvalue", type=float, default=5.0,
                        help="Target local significance to evaluate (default: 5.0)")
    args = parser.parse_args()

    trigger = args.trigger.lower()
    target_Z = args.ExpectedLocalZvalue
    
    methods = ["naive", "linear", "copula", "poisson_event", "decorrelated_bootstrap"]
    methods = ["naive", "copula", "poisson_event"]
    colors = {"naive": "red", "linear": "blue", "copula": "orange",
              "poisson_event": "green", "exclusive_categories": "purple",
              "decorrelated_bootstrap": "olive"}
              
    method_label_map = {"naive": "Independent", "linear": "Overlap", "copula": "Copula",
                        "poisson_event": "Poisson Bootstrap", "decorrelated_bootstrap": "Decorrelated Bootstrap"}
    
    os.makedirs("plots", exist_ok=True)

    print(f"\n{'='*65}")
    print(f" TRIGGER-WIDE GLOBAL SIGNIFICANCE ({trigger.upper()})")
    print(f"{'='*65}")

    plt.figure(figsize=(10, 6))

    for method in methods:
        # 1. Load data for the specific trigger
        # Looking for GP files or falling back to merged legacy files
        file_list = glob.glob(f"results/global_stat_{trigger}_{method}_*.npy")
        file_list = None
        if not file_list:
            file_list = glob.glob(f"results/merged15/final_{trigger}_{method}.npy")
            
        if not file_list:
            print(f"[{method.upper()}] Missing data for {trigger}. Skipping.")
            continue
            
        # Combine all files found for this method
        arr = np.concatenate([np.load(f) for f in file_list])
        trigger_t_max = arr[np.isfinite(arr)]
        n_toys = len(trigger_t_max)
        
        if n_toys == 0:
            continue
            
        # 2. Convert Test Statistic to Local Z for plotting the X-axis
        # t = -ln(p) -> p = exp(-t)
        p_local_dist = np.exp(-trigger_t_max)
        p_local_dist = np.clip(p_local_dist, 1e-300, 0.999999)
        z_local_dist = stats.norm.isf(p_local_dist)

        # 3. Calculate Global Statistics for the Target Z
        NrFound = np.sum(z_local_dist >= target_Z)
        p_global = NrFound / n_toys
        Z_global = stats.norm.isf(p_global) if p_global > 0 else np.inf

        print(f"\n###### RESULT: {method.upper()} ######")
        print(f" Number of pseudo-experiments = {n_toys}")
        print(f" Toys with Local Z >= {target_Z} = {NrFound}")
        if p_global > 0:
            print(f" Trigger-Wide Global p-value = {p_global:.2e} (Global Z = {Z_global:.2f})")
        else:
            print(f" Trigger-Wide Global p-value = < {1/n_toys:.2e} (Need more toys)")
            
        # 4. Plot the Survival Curve (Global vs Local)
        z_local_sorted = np.sort(z_local_dist)[::-1]
        ranks = np.arange(1, n_toys + 1)
        p_global_curve = ranks / n_toys
        z_global_curve = stats.norm.isf(p_global_curve)

        valid = (z_global_curve > -10) & np.isfinite(z_global_curve)
        method_label = method_label_map.get(method, method.capitalize())
        
        plt.plot(z_local_sorted[valid], z_global_curve[valid],
                 label=f"{method_label} (N={n_toys})", color=colors.get(method, 'black'), lw=2)

    # 5. Format the Plot
    plt.title(f"Trigger-Wide Global Significance vs. BumpHunter Significance\nTrigger: {trigger.upper()} | GP Background Model", fontsize=14)
    plt.xlabel(f"Highest Observed Local Significance in {trigger.upper()} ($Z_{{BH}}$)", fontsize=12)
    plt.ylabel("Trigger-Wide Global Significance ($Z_{global}$)", fontsize=12)
    
    plt.axhline(3, color='grey', linestyle='--', alpha=0.7, label='3σ Global Evidence')
    plt.axhline(5, color='black', linestyle=':', alpha=0.7, label='5σ Global Discovery')
    
    lims = [max(0, plt.xlim()[0]), min(8, plt.xlim()[1])]
    plt.plot(lims, lims, 'k--', alpha=0.3, label="No LEE ($Z_{global} = Z_{BH}$)")

    plt.legend(loc="lower right")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    plot_out = f"plots/Trigger_Wide_Global_Z_{trigger}_GP.png"
    plt.savefig(plot_out, dpi=300)
    print(f"\n{'-'*65}\nPlot saved to {plot_out}\n")

if __name__ == "__main__":
    main()
