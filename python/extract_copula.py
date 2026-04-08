import ROOT
import numpy as np
from scipy.stats import rankdata
import sys

def extract_copula(input_root, output_npz):
    print(f"Reading TTree from {input_root}...")
    df = ROOT.RDataFrame("output", input_root)
    mass_vars = ["Mjj", "Mbb", "Mjb", "Mee", "Mmm", "Mje", "Mjm", "Mjg", "Mbe", "Mbm", "Mbg"]
    
    try:
        data_dict = df.AsNumpy(columns=mass_vars)
    except Exception as e:
        print(f"Failed to read columns: {e}")
        return

    N = len(data_dict["Mjj"])
    print(f"Loaded {N} events.")
    copula_matrix = np.zeros((N, len(mass_vars)))

    print("Converting valid masses to empirical CDF quantiles...")
    for i, var in enumerate(mass_vars):
        data = data_dict[var]
        
        # FIX: Find the physically valid masses (ignoring exact 0s or near-0 floats)
        valid_mask = data > 0.001 
        
        # Initialize the whole column to -1.0 (our "missing particle" flag)
        U = np.full(N, -1.0) 
        
        # Only rank the events that actually contain these particles
        valid_data = data[valid_mask]
        if len(valid_data) > 0:
            ranks = rankdata(valid_data)
            U[valid_mask] = ranks / (len(valid_data) + 1.0)
            
        copula_matrix[:, i] = U
        print(f"  {var}: {np.sum(valid_mask)} valid events.")

    np.savez(output_npz, copula=copula_matrix, columns=mass_vars)
    print(f"Successfully saved Copula matrix to {output_npz}")

if __name__ == "__main__":
    extract_copula(sys.argv[1], sys.argv[2])
