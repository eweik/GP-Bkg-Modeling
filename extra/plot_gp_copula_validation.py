#!/usr/bin/env python3
import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
import warnings

# Suppress SciPy constant array warnings and Sklearn convergence warnings globally
warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)
warnings.filterwarnings("ignore", module="sklearn.gaussian_process")

# Setup paths to import local modules
current_script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_script_dir)
if repo_root not in sys.path: sys.path.append(repo_root)
if os.getcwd() not in sys.path: sys.path.append(os.getcwd())

from src.config import ATLAS_BINS

def fit_gp_background(centers, density, density_err, min_len_scale_log=0.15):
    """Fits the Zero-Mean Gaussian Process in log-space."""
    mask = density > 0
    if np.sum(mask) < 3:
        return np.zeros_like(centers) # Failsafe for empty channels
        
    X_log = np.log(centers[mask]).reshape(-1, 1)
    y_target = np.log(density[mask])
    y_err_target = density_err[mask] / density[mask]

    # Physics-driven kernel constraints
    kernel = C(1.0, (1e-3, 1e2)) * RBF(
        length_scale=max(min_len_scale_log * 2.0, 1e-4), 
        length_scale_bounds=(min_len_scale_log, 5.0)
    )
    
    gp = GaussianProcessRegressor(kernel=kernel, alpha=y_err_target**2, n_restarts_optimizer=5, normalize_y=False)
    gp.fit(X_log, y_target)
        
    X_full_log = np.log(centers).reshape(-1, 1)
    y_pred_target = gp.predict(X_full_log)
    
    return np.exp(y_pred_target)

def get_channel_gp_data(base_dir, trigger, channel, cms, mass_matrix, col_names, min_len):
    """Extracts raw data, fits the GP, and calculates the discrete CDF & bounds."""
    fitfile = os.path.join(base_dir, "fits", f"fitme_p5_{trigger}_{channel}.json")
    if not os.path.exists(fitfile):
        return None
        
    with open(fitfile, "r") as j_nom:
        d_nom = json.load(j_nom)
        
    fmin_val, fmax_val = float(d_nom['fmin']), float(d_nom['fmax'])
    v_bins = ATLAS_BINS[(ATLAS_BINS >= fmin_val) & (ATLAS_BINS <= fmax_val)]
    c = (v_bins[:-1] + v_bins[1:]) / 2
    widths = np.diff(v_bins)
    
    # Extract Raw Data for GP fitting
    idx = col_names.index(f"M{channel}")
    masses = mass_matrix[:, idx]
    valid_masses = masses[masses > 0] * cms
    data_counts, _ = np.histogram(valid_masses, bins=v_bins)
    
    bkg_density = data_counts / widths
    bkg_err_density = np.sqrt(np.maximum(data_counts, 1.0)) / widths
    
    # 1. Evaluate GP Fit
    expected_density = fit_gp_background(c, bkg_density, bkg_err_density, min_len)
    expected_counts = expected_density * widths
    
    # Generate high-density curve for smooth plotting
    x_dense = np.linspace(v_bins[0], v_bins[-1], 1000)
    dense_widths = np.diff(np.append(x_dense, x_dense[-1] + (x_dense[1]-x_dense[0])))
    gp_dense_density = fit_gp_background(x_dense, bkg_density, bkg_err_density, min_len)
    y_dense = gp_dense_density * dense_widths * (len(c)/1000.0) # Scale roughly for visual plot
    
    # 2. Create Discrete CDF
    cdf = np.cumsum(expected_counts) / np.sum(expected_counts)

    # 3. Calculate Phase-Space Bounds
    if len(valid_masses) > 0:
        u_min = np.sum(valid_masses < v_bins[0]) / len(valid_masses)
        u_max = np.sum(valid_masses <= v_bins[-1]) / len(valid_masses)
    else:
        u_min, u_max = 0.0, 1.0
        
    return data_counts, expected_counts, cdf, v_bins, c, (u_min, u_max), x_dense, gp_dense_density

def map_uniform_to_mass(u_array, u_bounds, cdf, centers, apply_jitter=False):
    """Maps uniform [0,1] variables to discrete bins, with optional jitter."""
    u_min, u_max = u_bounds
    if apply_jitter: u_array = u_array + np.random.uniform(-0.0002, 0.0002, size=len(u_array))
    u_trunc = (u_array - u_min) / max(u_max - u_min, 1e-10)
    u_trunc = np.abs(u_trunc)
    u_trunc = np.where(u_trunc >= 1.0, 1.99999 - u_trunc, u_trunc)
    idx = np.searchsorted(cdf, u_trunc)
    return centers[np.clip(idx, 0, len(centers) - 1)]

def safe_pearson(x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0: return 0.0
    r, _ = stats.pearsonr(x, y)
    return 0.0 if np.isnan(r) else r

def safe_spearman(x, y):
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0: return 0.0
    s, _ = stats.spearmanr(x, y)
    return 0.0 if np.isnan(s) else s

def generate_expected_gp_marginal(copula_matrix, col_idx, u_bounds, cdf, centers, bins):
    """Generates expected toy marginal by averaging 1000 resamples."""
    N_data = len(copula_matrix)
    N_draw_total = N_data * 1000
    
    sampled_rows = copula_matrix[np.random.choice(N_data, size=N_draw_total, replace=True)]
    u_raw = sampled_rows[sampled_rows[:, col_idx] >= 0, col_idx]
    
    mask = (u_raw >= u_bounds[0]) & (u_raw <= u_bounds[1])
    u_in_window = u_raw[mask]
    
    if len(u_in_window) == 0: return np.zeros(len(bins) - 1)
        
    m_toy = map_uniform_to_mass(u_in_window, u_bounds, cdf, centers, apply_jitter=True)
    toy_counts_total, _ = np.histogram(m_toy, bins=bins)
    
    return toy_counts_total / 1000.0

def main():
    parser = argparse.ArgumentParser(description="GP Copula Validation: Marginals and Correlation Matrices")
    parser.add_argument('--trigger', type=str, required=True, help="Trigger name")
    parser.add_argument('--ch1', type=str, default='jj', help="First channel for marginal plot")
    parser.add_argument('--ch2', type=str, default='jb', help="Second channel for marginal plot")
    parser.add_argument('--cms', type=float, default=13000., help="Center of mass energy")
    parser.add_argument('--min-len', type=float, default=0.15, help="GP Min Length Scale")
    args = parser.parse_args()

    base_dir = os.getcwd() if os.path.exists("data") else repo_root
    
    print("1. Loading Data Matrices...")
    f_mass = np.load(os.path.join(base_dir, "data", f"masses_{args.trigger}.npz"))
    f_copula = np.load(os.path.join(base_dir, "data", f"copula_{args.trigger}.npz"))
    mass_matrix, copula_matrix = f_mass['masses'], f_copula['copula']
    col_names = list(f_mass['columns'])
    n_cols = len(col_names)

    print("2. Fitting Gaussian Processes and calculating phase-space bounds...")
    channel_info = {}
    for i, col in enumerate(col_names):
        ch = col.replace("M", "")
        data = get_channel_gp_data(base_dir, args.trigger, ch, args.cms, mass_matrix, col_names, args.min_len)
        if data:
            data_counts, expected_counts, cdf, v_bins, c, u_bounds, x_dense, gp_dense = data
            channel_info[i] = {
                'name': ch, 'cdf': cdf, 'centers': c, 'bins': v_bins, 'bounds': u_bounds,
                'data_counts': data_counts, 'gp_counts': expected_counts, 
                'x_dense': x_dense, 'gp_dense': gp_dense
            }

    # ==========================================
    # PLOT 1: MARGINAL AGREEMENT (1D)
    # ==========================================
    print(f"3. Generating GP Marginal Validation for {args.ch1.upper()} and {args.ch2.upper()}...")
    fig1, axes1 = plt.subplots(1, 2, figsize=(16, 7))
    
    for ax, ch_name in zip(axes1, [args.ch1, args.ch2]):
        idx = col_names.index(f"M{ch_name}")
        info = channel_info.get(idx)
        if not info: continue
        
        expected_toys = generate_expected_gp_marginal(
            copula_matrix, idx, info['bounds'], info['cdf'], info['centers'], info['bins']
        )
        
        # We plot density (Events / GeV) to properly show the GP continuous line
        widths = np.diff(info['bins'])
        data_density = info['data_counts'] / widths
        data_err = np.sqrt(np.maximum(info['data_counts'], 1.0)) / widths
        gp_density_pts = info['gp_counts'] / widths
        toy_density = expected_toys / widths
        
        # GP Continuous Line
        ax.plot(info['x_dense'], info['gp_dense'], color='dodgerblue', linewidth=2, label='Gaussian Process (Smooth Mean)', zorder=2)
        # GP Bin Center Expectations
        ax.plot(info['centers'], gp_density_pts, color='dodgerblue', marker='s', linestyle='none', markersize=5, zorder=3)
        # Copula Toys (Dots)
        ax.plot(info['centers'], toy_density, color='red', marker='o', linestyle='none', markersize=6, alpha=0.9, markeredgecolor='darkred', label='Copula Toys (Unscaled Average)', zorder=4)
        # Raw Data
        ax.errorbar(info['centers'], data_density, yerr=data_err, fmt='ko', markersize=4, capsize=3, label='Raw Data', zorder=10)

        ax.set_title(f"Marginal Agreement: {ch_name.upper()} Channel", fontsize=15)
        ax.set_xlabel(f"${ch_name.upper()}$ Mass [GeV]", fontsize=14)
        ax.set_ylabel("Events / GeV", fontsize=14)
        ax.set_yscale('log')
        ax.set_xlim(info['bins'][0], info['bins'][-1])
        ax.set_ylim(max(1e-5, np.min(data_density[data_density>0])*0.5), np.max(data_density)*5.0)
        ax.legend(fontsize=11, frameon=True, edgecolor='black')
        ax.grid(True, which="both", linestyle="--", alpha=0.3)

    plt.suptitle(f"GP Copula Marginal Validation | Trigger: {args.trigger.upper()} | Length Scale $\ell \geq {args.min_len}$", fontsize=18)
    fig1.tight_layout()
    plot_dir = os.path.join(base_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    fig1.savefig(os.path.join(plot_dir, f"gp_marginal_agreement_{args.trigger}_{args.ch1}_{args.ch2}.png"), dpi=300, bbox_inches='tight')

    # ==========================================
    # PLOT 2 & 3: CORRELATION MATRICES (2D)
    # ==========================================
    print("4. Mapping Massive Toy Sample for Correlation Matrices...")
    N_toys = 200000 
    corr_p_raw, corr_s_raw = np.eye(n_cols), np.eye(n_cols)
    corr_p_cop, corr_s_cop = np.eye(n_cols), np.eye(n_cols)
    
    for i in range(n_cols):
        for j in range(i + 1, n_cols):
            if i not in channel_info or j not in channel_info: continue
            info_i, info_j = channel_info[i], channel_info[j]
            
            # --- RAW DATA (In-Window) ---
            m_i_all, m_j_all = mass_matrix[:, i] * args.cms, mass_matrix[:, j] * args.cms
            b_i, b_j = (info_i['bins'][0], info_i['bins'][-1]), (info_j['bins'][0], info_j['bins'][-1])
            
            mask_raw = (m_i_all >= b_i[0]) & (m_i_all <= b_i[1]) & (m_j_all >= b_j[0]) & (m_j_all <= b_j[1])
            m_i_raw, m_j_raw = m_i_all[mask_raw], m_j_all[mask_raw]
            
            corr_p_raw[i, j] = corr_p_raw[j, i] = safe_pearson(m_i_raw, m_j_raw)
            corr_s_raw[i, j] = corr_s_raw[j, i] = safe_spearman(m_i_raw, m_j_raw)

            # --- COPULA TOYS ---
            rand_idx = np.random.choice(len(copula_matrix), size=N_toys, replace=True)
            u_i_samp, u_j_samp = copula_matrix[rand_idx, i], copula_matrix[rand_idx, j]
            
            surv = (u_i_samp >= info_i['bounds'][0]) & (u_i_samp <= info_i['bounds'][1]) & \
                   (u_j_samp >= info_j['bounds'][0]) & (u_j_samp <= info_j['bounds'][1])
                   
            u_i_surv, u_j_surv = u_i_samp[surv], u_j_samp[surv]
            
            m_i_toy = map_uniform_to_mass(u_i_surv, info_i['bounds'], info_i['cdf'], info_i['centers'], True)
            m_j_toy = map_uniform_to_mass(u_j_surv, info_j['bounds'], info_j['cdf'], info_j['centers'], True)
            
            corr_p_cop[i, j] = corr_p_cop[j, i] = safe_pearson(m_i_toy, m_j_toy)
            corr_s_cop[i, j] = corr_s_cop[j, i] = safe_spearman(m_i_toy, m_j_toy)

    def plot_matrix_pair(raw_mat, cop_mat, metric_name, filename):
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        cmap = sns.diverging_palette(220, 20, as_cmap=True)
        sns.heatmap(raw_mat, ax=axes[0], cmap=cmap, vmin=-0.1, vmax=1.0, xticklabels=col_names, yticklabels=col_names, annot=True, fmt=".2f", square=True)
        axes[0].set_title(f"Raw Data {metric_name} $\\rho$ (In-Window)\n{args.trigger.upper()}", fontsize=14)
        sns.heatmap(cop_mat, ax=axes[1], cmap=cmap, vmin=-0.1, vmax=1.0, xticklabels=col_names, yticklabels=col_names, annot=True, fmt=".2f", square=True)
        axes[1].set_title(f"Copula Toys {metric_name} $\\rho$ (Mapped to GP)\n{args.trigger.upper()}", fontsize=14)
        plt.suptitle(f"Validation of {metric_name} Correlation Preservation (GP Marginal)", fontsize=18, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(plot_dir, filename), dpi=300, bbox_inches='tight')
        plt.close(fig)

    print("5. Plotting Heatmaps...")
    plot_matrix_pair(corr_p_raw, corr_p_cop, "Linear (Pearson)", f"gp_pearson_matrix_{args.trigger}.png")
    plot_matrix_pair(corr_s_raw, corr_s_cop, "Rank (Spearman)", f"gp_spearman_matrix_{args.trigger}.png")
    
    print("All GP Validation plots successfully generated!")

if __name__ == "__main__":
    main()
