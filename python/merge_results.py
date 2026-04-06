#!/usr/bin/env python3
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import argparse

def main(args):
    os.makedirs("results/merged", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("plots/LEE_test_statistics", exist_ok=True)
    
    triggers = [f"t{i}" for i in range(1, 8)] if args.trigger == "all" else [args.trigger]
    methods = ["naive", "linear", "copula", "poisson_event", "decorrelated_bootstrap"]
    
    # Standard 1-sided Z-score p-values
    # 1 sigma = 0.15865, 2 sigma = 0.02275, 3 sigma = 0.00135, 4 sigma = 3.167e-05
    z_scores = [1, 2, 3, 4]
    p_values = [norm.sf(z) for z in z_scores]
    percentiles = [(1.0 - p) * 100 for p in p_values]

    for trig in triggers:
        print(f"\n{'='*50}\nProcessing Trigger: {trig}\n{'='*50}")
        
        plt.figure(figsize=(10, 6))
        colors = {"naive": "red", "linear": "blue", "copula": "orange",
                  "poisson_event": "green", "exclusive_categories": "purple",
                  "decorrelated_bootstrap": "olive"}
        
        for method in methods:
            # Find all chunked arrays for this trigger and method
            search_pattern = f"results/global_stat_{trig}_{method}_*.npy"
            file_list = glob.glob(search_pattern)
            
            if not file_list:
                print(f"[{method}] No files found.")
                continue
                
            # Load and concatenate
            arrays = [np.load(f) for f in file_list]
            merged_data = np.concatenate(arrays)
            
            # Clean out any rare NaNs or Infs just in case
            merged_data = merged_data[np.isfinite(merged_data)]
            total_toys = len(merged_data)
            
            # Save the stitched array so you can delete the thousands of small chunks later
            merged_out = f"results/merged/final_{trig}_{method}.npy"
            np.save(merged_out, merged_data)
            
            print(f"[{method.upper()}] Merged {len(file_list)} files -> {total_toys} total toys.")
            
            # Calculate Global Significance Thresholds
            if total_toys > 0:
                print(f"  Mean t_max: {np.mean(merged_data):.2f}")
                print("  Global Significance Thresholds (t_max required):")
                
                thresholds = np.percentile(merged_data, percentiles)
                for z, thresh in zip(z_scores, thresholds):
                    # Only print if we actually have enough toys to resolve this p-value
                    if total_toys >= (1.0 / norm.sf(z)):
                        print(f"    {z} sigma global -> t_max > {thresh:.2f}")
                    else:
                        print(f"    {z} sigma global -> Need more toys to resolve empirically.")
                        
                # Plotting the distribution
                plt.hist(merged_data, bins=100, range=(0, max(50, np.max(merged_data))), 
                         histtype='step', linewidth=2, label=f"{method.capitalize()} (N={total_toys})",
                         color=colors[method], density=True, log=True)

        # Finalize and save the plot
        plt.title(f"Global t_max Distributions for Trigger {trig} (Look-Elsewhere Effect)")
        plt.xlabel(r"Maximum Local Test Statistic ($t_{max}$)")
        plt.ylabel("Probability Density")
        plt.legend(loc="upper right")
        plt.grid(True, which="both", linestyle="--", alpha=0.5)
        plt.tight_layout()
        
        plot_out = f"plots/LEE_test_statistics/LEE_Distributions_{trig}.png"
        plt.savefig(plot_out, dpi=300)
        plt.close()
        print(f"\nSaved plot to {plot_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge toys and compute LEE.")
    parser.add_argument("--trigger", type=str, default="all", help="Trigger to merge (e.g., t1) or 'all'")
    args = parser.parse_args()
    main(args)

