import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Plot Global Significance Test Statistics")
    parser.add_argument('--trigger', type=str, default='t2', help="Trigger stream (e.g., t1, t2)")
    args = parser.parse_args()

    trigger = args.trigger.lower()

    methods = {
        "naive": {"color": "red", "label": "Naive (Independent)"},
        # "linear": {"color": "orange", "label": "Linear (Bin Locked)"},
        "copula": {"color": "green", "label": "Empirical Copula (Migrated)"},
        "poisson_event": {"color": "blue", "label": "Poisson Bootstrap"},
        "decorrelated_bootstrap": {"color": "orange", "label": "Decorrelated Bootstrap"}
    }
                
    plt.figure(figsize=(10, 7))

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    plot_dir = os.path.join(base_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    for m, settings in methods.items():
        filename = os.path.join(results_dir, f"global_stat_{trigger}_{m}_zero_local.npy")
        
        if not os.path.exists(filename):
            print(f"Warning: Data file {filename} not found. Skipping {m}.")
            continue
            
        data = np.load(filename)
        
        plt.hist(data, bins=50, histtype='step',
                 label=f"{settings['label']}",
                 color=settings['color'], linewidth=2.5, density=True)

    plt.yscale('log')
    plt.xlabel(r'Global Test Statistic ($t_{max}$)', fontsize=16)
    plt.ylabel('Probability Density', fontsize=16)
    plt.title(f'Global Significance Comparison - Trigger {trigger.upper()} - GP Background Model', fontsize=18)
    plt.legend(fontsize=11, frameon=False, loc='upper right')

    out_filename = os.path.join(plot_dir, f"global_significance_impact_{trigger}_gp.png")
    plt.savefig(out_filename, dpi=300)
    print(f"Plot successfully saved to {out_filename}.")

if __name__ == "__main__":
    main()
