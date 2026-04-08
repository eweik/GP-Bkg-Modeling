import ROOT
import numpy as np
import sys

def extract_masses(input_root, output_npz):
    print(f"Reading TTree from {input_root}...")
    df = ROOT.RDataFrame("output", input_root)
    mass_vars = ["Mjj", "Mbb", "Mjb", "Mje", "Mjm", "Mjg", "Mbe", "Mbm", "Mbg"]
    
    try:
        data_dict = df.AsNumpy(columns=mass_vars)
    except Exception as e:
        print(f"Failed to read columns: {e}")
        return

    N = len(data_dict["Mjj"])
    print(f"Loaded {N} events.")
    mass_matrix = np.zeros((N, len(mass_vars)))

    print("Filtering and saving valid physical masses...")
    for i, var in enumerate(mass_vars):
        data = data_dict[var]
        
        # Find the physically valid masses (ignoring exact 0s or near-0 floats)
        valid_mask = data > 0.001 
        
        # Initialize the whole column to -1.0 (our "missing particle" flag)
        M = np.full(N, -1.0) 
        
        # Only assign the raw mass values to events that passed the respective cuts
        M[valid_mask] = data[valid_mask]
            
        mass_matrix[:, i] = M
        print(f"  {var}: {np.sum(valid_mask)} valid events.")

    # Note: The key is now 'masses' instead of 'copula'
    np.savez(output_npz, masses=mass_matrix, columns=mass_vars)
    print(f"Successfully saved Mass matrix to {output_npz}")

if __name__ == "__main__":
    extract_masses(sys.argv[1], sys.argv[2])
