#!/usr/bin/env python3
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Calculate Experiment-Wide Global Significance.")
    parser.add_argument("--ExpectedLocalZvalue", type=float, default=5.0,
                        help="Target local significance to evaluate (default: 5.0)")
    args = parser.parse_args()

    target_Z = args.ExpectedLocalZvalue
    methods = ["naive", "linear", "copula"]
    colors = {"naive": "red", "linear": "blue", "copula": "green"}
    # methods = ["naive", "linear"]
    # colors = {"naive": "red", "linear": "blue"}
    
    os.makedirs("plots", exist_ok=True)

    print(f"\n{'='*65}")
    print(f" ANALYSIS-WIDE GLOBAL SIGNIFICANCE (All 7 Triggers)")
    print(f"{'='*65}")

    plt.figure(figsize=(10, 6))

    for method in methods:
        trigger_arrays = []
        
        # 1. Load data for all 7 triggers
        for t in range(1, 8):
            trigger = f"t{t}"
            file_list = glob.glob(f"results/global_stat_{trigger}_{method}_*.npy")
            if not file_list:
                file_list = glob.glob(f"results/merged/final_{trigger}_{method}.npy")
                
            if not file_list:
                print(f"[{method.upper()}] Missing data for {trigger}. Cannot compute experiment-wide.")
                trigger_arrays = []
                break
                
            arr = np.concatenate([np.load(f) for f in file_list])
            arr = arr[np.isfinite(arr)]
            trigger_arrays.append(arr)
            
        if not trigger_arrays:
            continue
            
        # 2. Align the arrays to the minimum number of toys
        min_toys = min(len(a) for a in trigger_arrays)
        aligned_arrays = [a[:min_toys] for a in trigger_arrays]
        
        # 3. Compute the Experiment-Wide Maximum Test Statistic
        # This is the core math: max(T1, T2, ..., T7)
        experiment_t_max = np.max(aligned_arrays, axis=0)
        
        # 4. Convert Test Statistic to Local Z for plotting the X-axis
        # t = -ln(p) -> p = exp(-t)
        p_local_dist = np.exp(-experiment_t_max)
        p_local_dist = np.clip(p_local_dist, 1e-300, 0.999999)
        z_local_dist = stats.norm.isf(p_local_dist)

        # 5. Calculate Global Statistics for the Target Z
        NrFound = np.sum(z_local_dist >= target_Z)
        p_global = NrFound / min_toys
        Z_global = stats.norm.isf(p_global) if p_global > 0 else np.inf

        print(f"\n###### RESULT: {method.upper()} ######")
        print(f" Number of pseudo-experiments = {min_toys}")
        print(f" Toys with Local Z >= {target_Z} = {NrFound}")
        if p_global > 0:
            print(f" Analysis-Wide Global p-value = {p_global:.2e} (Global Z = {Z_global:.2f})")
        else:
            print(f" Analysis-Wide Global p-value = < {1/min_toys:.2e} (Need more toys)")
            
        # 6. Plot the Survival Curve (Global vs Local)
        z_local_sorted = np.sort(z_local_dist)[::-1]
        ranks = np.arange(1, min_toys + 1)
        p_global_curve = ranks / min_toys
        z_global_curve = stats.norm.isf(p_global_curve)

        valid = (z_global_curve > -10) & np.isfinite(z_global_curve)
        plt.plot(z_local_sorted[valid], z_global_curve[valid],
                 label=f"{method.capitalize()} (N={min_toys})", color=colors[method], lw=2)

    # 7. Format the Plot
    plt.title("Analysis-Wide Global Significance vs. BumpHunter Significance - GP Background Model", fontsize=14)
    plt.xlabel("Highest Observed Local Significance Across All Triggers ($Z_{BH}$)", fontsize=12)
    plt.ylabel("Analysis-Wide Global Significance ($Z_{global}$)", fontsize=12)
    
    plt.axhline(3, color='grey', linestyle='--', alpha=0.7, label='3σ Global Evidence')
    plt.axhline(5, color='black', linestyle=':', alpha=0.7, label='5σ Global Discovery')
    
    lims = [max(0, plt.xlim()[0]), min(8, plt.xlim()[1])]
    plt.plot(lims, lims, 'k--', alpha=0.3, label="No LEE ($Z_{global} = Z_{BH}$)")

    plt.legend(loc="lower right")
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    plt.tight_layout()
    
    plot_out = "plots/Analysis_Wide_Global_Z.png"
    plt.savefig(plot_out, dpi=300)
    print(f"\n{'-'*65}\nMaster plot saved to {plot_out}\n")

if __name__ == "__main__":
    main()
