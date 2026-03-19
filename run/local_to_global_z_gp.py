#!/usr/bin/env python3
import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Map Local Z to Global Z.")
    parser.add_argument("--ExpectedLocalZvalue", type=float, default=5.0,
                        help="Expected local significance (default: 5.0)")
    parser.add_argument("--trigger", type=str, default="t1",
                        help="Trigger to analyze (default: t1)")
    args = parser.parse_args()

    ExpectedLocalZvalue = args.ExpectedLocalZvalue
    trigger = args.trigger.lower()

    methods = ["naive", "linear", "copula"]
    colors = {"naive": "red", "linear": "blue", "copula": "green"}

    os.makedirs("plots", exist_ok=True)

    print(f"\n############## START ################")
    print(f"Mapping Local Z to Global Z (Look-Elsewhere Effect)")
    print(f"Trigger: {trigger.upper()} | Expected Local Z: {ExpectedLocalZvalue}")

    plt.figure(figsize=(10, 6))

    for method in methods:
        # 1. Load the generated data (Looking for the new GP files)
        file_list = glob.glob(f"results/global_stat_GP_{trigger}_{method}_*.npy")
        if not file_list:
            # Fallback if merged
            file_list = glob.glob(f"results/merged/final_GP_{trigger}_{method}.npy")
            if not file_list:
                print(f"Warning: No toy files found for {method.upper()}")
                continue

        arrays = [np.load(f) for f in file_list]
        t_max_dist = np.concatenate(arrays)
        t_max_dist = t_max_dist[np.isfinite(t_max_dist)]
        MaxEvents = len(t_max_dist)

        if MaxEvents == 0: continue

        # 2. Convert BumpHunter test statistic (t) to Local Z-score
        # t = -ln(local_pvalue)  =>  local_pvalue = exp(-t)
        p_local_dist = np.exp(-t_max_dist)
        
        # Prevent perfectly zero p-values from float precision
        p_local_dist = np.clip(p_local_dist, 1e-300, 0.999999)
        z_local_dist = stats.norm.isf(p_local_dist)

        # 3. Calculate Global p-value for the requested ExpectedLocalZvalue
        NrFound = np.sum(z_local_dist >= ExpectedLocalZvalue)
        p_global = NrFound / MaxEvents
        
        if p_global > 0:
            Z_global = stats.norm.isf(p_global)
        else:
            Z_global = np.inf 

        # 4. Print Output Block
        print(f"\n###### RESULT: {method.upper()} ######")
        print(f" Total pseudo-experiments = {MaxEvents}")
        print(f" Toys with Local Z >= {ExpectedLocalZvalue} = {NrFound}")
        if p_global > 0:
            print(f" Global p-value = {p_global:.2e}  ->  Global Z = {Z_global:.2f}")
        else:
            print(f" Global p-value = < {1/MaxEvents:.2e} (Not enough toys for this Z)")

        # 5. Prepare the Plot
        z_local_sorted = np.sort(z_local_dist)[::-1] # Sort highest Z first
        ranks = np.arange(1, MaxEvents + 1)
        p_global_curve = ranks / MaxEvents
        z_global_curve = stats.norm.isf(p_global_curve)

        valid = (z_global_curve > -10) & np.isfinite(z_global_curve)
        plt.plot(z_local_sorted[valid], z_global_curve[valid],
                 label=f"{method.capitalize()} (N={MaxEvents:,})", color=colors[method], lw=2)

    # 6. Format the Plot
    plt.title(f"Look-Elsewhere Effect: Local vs. Global Significance ({trigger.upper()})", fontsize=14, fontweight='bold')
    plt.xlabel("Observed Local Significance ($Z_{local}$)", fontsize=13)
    plt.ylabel("Global Significance ($Z_{global}$)", fontsize=13)
    
    # Add standard discovery thresholds
    plt.axhline(3, color='grey', linestyle='--', alpha=0.7, label='3$\sigma$ Global Evidence')
    plt.axhline(5, color='black', linestyle=':', alpha=0.7, label='5$\sigma$ Global Discovery')

    # Add the "No Look-Elsewhere Effect" baseline (Z_global = Z_local)
    lims = [max(0, plt.xlim()[0]), min(8, plt.xlim()[1])]
    plt.plot(lims, lims, 'k--', alpha=0.3, label="No LEE ($Z_{global} = Z_{local}$)")

    plt.legend(loc="lower right", fontsize=11, framealpha=0.9)
    plt.grid(True, which="both", linestyle="--", alpha=0.5)
    
    # Ensure standard axis boundaries for comparison
    plt.xlim(0, 8)
    plt.ylim(0, 8)
    plt.tight_layout()
    
    plot_out = f"plots/LEE_Curve_GP_{trigger}.png"
    plt.savefig(plot_out, dpi=300)
    print(f"\n###### END RESULT ######")
    print(f"Plot saved to {plot_out}\n")

if __name__ == "__main__":
    main()
