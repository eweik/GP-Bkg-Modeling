#!/usr/bin/env python3
import os
import sys
import json
import time
import numpy as np
import warnings
from argparse import ArgumentParser
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

current_script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_script_dir)
if repo_root not in sys.path:
    sys.path.append(repo_root)
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from src.config import ATLAS_BINS, TRIGGER_OVERLAPS
from src.models import FiveParam
from src.stats import fast_bumphunter_stat

def fit_gp_background(centers, density, density_err, parametric_density, min_len_scale_log, gp_mean_type):
    """
    Fits the GP once to establish the frozen, expected background.
    """
    mask = density > 0
    X_log = np.log(centers[mask]).reshape(-1, 1)
    
    if gp_mean_type == 'zero':
        # 0-mean function: GP learns the entire log-density directly
        y_target = np.log(density[mask])
    else:
        # 5-param mean function: GP learns the residual away from the analytic fit
        y_target = np.log(density[mask] / np.maximum(parametric_density[mask], 1e-15))

    y_err_target = density_err[mask] / density[mask]

    # Correct, physics-driven kernel:
    kernel = C(1.0, (1e-3, 1e2)) * RBF(
        length_scale=max(min_len_scale_log * 2.0, 0.5),
        length_scale_bounds=(min_len_scale_log, 5.0)
    )
    gp = GaussianProcessRegressor(kernel=kernel, alpha=y_err_target**2, n_restarts_optimizer=5, normalize_y=False)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X_log, y_target)
        
    X_full_log = np.log(centers).reshape(-1, 1)
    y_pred_target = gp.predict(X_full_log)
    
    if gp_mean_type == 'zero':
        return np.exp(y_pred_target)
    else:
        return parametric_density * np.exp(y_pred_target)

def main(args):
    os.makedirs("results", exist_ok=True)
    base_dir = os.getcwd() if os.path.exists("data") and os.path.exists("fits") else repo_root

    mass_types = ["jj", "bb", "jb", "je", "jm", "jg", "be", "bm", "bg"]
    bkg_expectations, channel_info = {}, {}
    overlap_map = TRIGGER_OVERLAPS.get(args.trigger.lower(), TRIGGER_OVERLAPS["default"])
    
    print(f"Initializing Frozen Baselines for {args.trigger.upper()} | GP Mean: {args.gp_mean}")

    # --- PRE-LOAD RAW DATA FOR GP FITTING & TOY LOGIC ---
    mass_path = os.path.join(base_dir, "data", f"masses_{args.trigger}.npz")
    if not os.path.exists(mass_path):
        print(f"Error: Mass data not found at {mass_path}"); sys.exit(1)
        
    f_mass = np.load(mass_path)
    mass_matrix, col_names = f_mass['masses'], list(f_mass['columns'])
    N_events = len(mass_matrix)

    # 1. Build and Freeze the Expected Backgrounds
    for m in mass_types:
        fitfile_nom = os.path.join(base_dir, "fits", f"fitme_p5_{args.trigger}_{m}.json")
        if not os.path.exists(fitfile_nom):
            continue
            
        try:
            with open(fitfile_nom, "r") as j_nom:
                d_nom = json.load(j_nom)

            fmin_val, fmax_val = float(d_nom['fmin']), float(d_nom['fmax'])
            v_bins = ATLAS_BINS[(ATLAS_BINS >= fmin_val) & (ATLAS_BINS <= fmax_val)]
            c = (v_bins[:-1] + v_bins[1:]) / 2
            widths = np.diff(v_bins)

            # Baseline analytic expectation
            counts_nom = FiveParam(args.cms, c, *[float(p) for p in d_nom['parameters']]) 

            if np.sum(counts_nom) > 0:
                channel_info[m] = {'centers': c, 'bins': v_bins, 'widths': widths}
                
                if args.fit:
                    # Extract RAW data to train the GP
                    idx = col_names.index(f"M{m}")
                    valid_masses = mass_matrix[mass_matrix[:, idx] > 0, idx] * args.cms
                    raw_counts, _ = np.histogram(valid_masses, bins=v_bins)
                    
                    bkg_density = raw_counts / widths
                    bkg_err_density = np.sqrt(np.maximum(raw_counts, 1.0)) / widths
                    parametric_density = counts_nom / widths
                    
                    smoothed_density = fit_gp_background(
                        c, bkg_density, bkg_err_density, parametric_density, args.min_len, args.gp_mean
                    )
                    bkg_expectations[m] = smoothed_density * widths
                else:
                    bkg_expectations[m] = counts_nom
                
        except Exception as e:
            print(f"Failed to load baseline for {m}: {e}")
            continue

    if not bkg_expectations:
        print("Error: No valid background baselines established."); sys.exit(1)

    # 2. Copula & Matrix Setup
    u_bounds = {}
    if args.method == "copula":
        copula_path = os.path.join(base_dir, "data", f"copula_{args.trigger}.npz")
        f = np.load(copula_path)
        matrix, col_names_cop = f['copula'], list(f['columns'])

        # CDFs are strictly built from the frozen background
        cdfs = {m: np.cumsum(b) / np.sum(b) for m, b in bkg_expectations.items()}

        # Find exact U-boundaries using the pre-loaded mass matrix
        for m, b in bkg_expectations.items():
            idx = col_names.index(f"M{m}")
            masses = mass_matrix[:, idx]
            valid_masses = masses[masses > 0] * args.cms

            fmin_val = channel_info[m]['bins'][0]
            fmax_val = channel_info[m]['bins'][-1]

            N_valid = len(valid_masses)
            if N_valid > 0:
                u_min = np.sum(valid_masses < fmin_val) / N_valid
                u_max = np.sum(valid_masses <= fmax_val) / N_valid
            else:
                u_min, u_max = 0.0, 1.0

            u_bounds[m] = (u_min, u_max)

    if args.method == "exclusive_categories":
        valid_mask = mass_matrix > 0
        powers = 2 ** np.arange(len(col_names))
        event_patterns = valid_mask.dot(powers)
        unique_patterns = np.unique(event_patterns)
        pattern_indices = {p: np.where(event_patterns == p)[0] for p in unique_patterns}

    stats = []
    attempts = 0
    max_attempts = args.toys * 50 
    
    fit_status_msg = "USING GP-SMOOTHED FROZEN NULL" if args.fit else "USING ANALYTIC FROZEN NULL"
    print(f"\nGenerating {args.toys} {args.method} toys | {fit_status_msg}...")
    start_time = time.time()
    
    # 3. Toy Loop (Now strictly evaluating, no fitting)
    while len(stats) < args.toys and attempts < max_attempts:
        attempts += 1
        toy_max_t = 0.0
        
        completed = len(stats)
        if completed > 0 and completed % max(1, (args.toys // 20)) == 0:
            progress = int((completed / args.toys) * 100)
            sys.stdout.write(f"\rProgress: [{('=' * (progress//5)).ljust(20)}] {progress}% (Attempts: {attempts}) ")
            sys.stdout.flush()

        if args.method == "naive":
            for m, b in bkg_expectations.items():
                toy_counts = np.random.poisson(b)
                toy_max_t = max(toy_max_t, fast_bumphunter_stat(toy_counts, b))
        
        elif args.method == "linear":
            jj_b = bkg_expectations.get('jj', None)
            jj_centers = channel_info['jj']['centers'] if jj_b is not None else None
            if jj_b is not None:
                jj_pseudo = np.random.poisson(jj_b)

            for m, b in bkg_expectations.items():
                if m == 'jj':
                    toy_counts = jj_pseudo
                else:
                    if jj_b is None:
                        toy_counts = np.random.poisson(b)
                    else:
                        ov_frac = overlap_map.get(m, 0.1)
                        m_centers = channel_info[m]['centers']

                        jj_b_aligned = np.zeros(len(b))
                        jj_pseudo_int = np.zeros(len(b), dtype=int)

                        for i, mc in enumerate(m_centers):
                            dist = np.abs(jj_centers - mc)
                            min_idx = np.argmin(dist)
                            if dist[min_idx] < 1.0:
                                jj_b_aligned[i] = jj_b[min_idx]
                                jj_pseudo_int[i] = jj_pseudo[min_idx]

                        expected_overlap = np.minimum(b * ov_frac, jj_b_aligned)
                        safe_jj_b = np.maximum(jj_b_aligned, 1e-15)
                        fluctuated_overlap = expected_overlap * (jj_pseudo_int / safe_jj_b)
                        expected_remaining = np.maximum(0.0, b - expected_overlap)
                        ind_counts = np.random.poisson(expected_remaining)
                        toy_counts = fluctuated_overlap + ind_counts

                toy_max_t = max(toy_max_t, fast_bumphunter_stat(toy_counts, b))

        elif args.method == "copula":
            # 1. Sample N times globally from the full dataset size
            N_draw = np.random.poisson(len(matrix))
            sampled_rows = matrix[np.random.choice(len(matrix), size=N_draw, replace=True)]
            
            for m, b in bkg_expectations.items():
                idx = col_names_cop.index(f"M{m}")
                
                # 2. Extract raw uniform copula values
                u_raw = sampled_rows[sampled_rows[:, idx] >= 0, idx]
                
                # 3. Apply phase-space cuts in uniform space
                u_min, u_max = u_bounds[m]
                mask_in_window = (u_raw >= u_min) & (u_raw <= u_max)
                u_in_window = u_raw[mask_in_window]
                
                if len(u_in_window) == 0:
                    toy_counts = np.zeros(len(b), dtype=int)
                else:
                    # Jitter to break ties from empirical extraction
                    u_jittered = u_in_window + np.random.uniform(-0.0002, 0.0002, size=len(u_in_window))
                    
                    # 4. Transform to strictly local truncated [0, 1) space
                    u_trunc = (u_jittered - u_min) / max(u_max - u_min, 1e-10)
                    
                    # Bound reflections for safety
                    u_trunc = np.abs(u_trunc)
                    u_trunc = np.where(u_trunc >= 1.0, 1.99999 - u_trunc, u_trunc)
                    
                    # 5. Map to physical binned mass via Inverse CDF
                    toy_counts = np.bincount(np.searchsorted(cdfs[m], u_trunc), minlength=len(b))
                
                toy_max_t = max(toy_max_t, fast_bumphunter_stat(toy_counts, b))

        elif args.method in ["poisson_event", "exclusive_categories", "decorrelated_bootstrap"]:
            
            if args.method == "decorrelated_bootstrap":
                N_draw = np.random.poisson(N_events)
                shuffled_matrix = np.copy(mass_matrix)
                for col_idx in range(shuffled_matrix.shape[1]):
                    np.random.shuffle(shuffled_matrix[:, col_idx])
                sampled_rows = shuffled_matrix[np.random.choice(N_events, size=N_draw, replace=True)]
                
            elif args.method == "poisson_event":
                N_draw = np.random.poisson(N_events)
                sampled_rows = mass_matrix[np.random.choice(N_events, size=N_draw, replace=True)]
                
            elif args.method == "exclusive_categories":
                sampled_rows_list = []
                for p, indices in pattern_indices.items():
                    n_obs = len(indices)
                    n_toy = np.random.poisson(n_obs)
                    if n_toy > 0:
                        sampled_rows_list.append(mass_matrix[np.random.choice(indices, size=n_toy, replace=True)])

                if len(sampled_rows_list) > 0:
                    sampled_rows = np.concatenate(sampled_rows_list, axis=0)
                else:
                    sampled_rows = np.empty((0, len(col_names)))

            for m, b in bkg_expectations.items():
                idx = col_names.index(f"M{m}")
                masses = sampled_rows[:, idx]
                valid_masses = masses[masses > 0] * args.cms
                toy_counts, _ = np.histogram(valid_masses, bins=channel_info[m]['bins'])

                if np.sum(toy_counts) < 50: continue

                toy_max_t = max(toy_max_t, fast_bumphunter_stat(toy_counts, b))

        stats.append(toy_max_t)

    sys.stdout.write(f"\rProgress: [{'=' * 20}] 100% \n")
    sys.stdout.flush()

    out_file = os.path.join("results", f"global_stat_{args.trigger}_{args.method}_{args.gp_mean}_{args.jobid}.npy")
    np.save(out_file, stats)
    
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    
    print("\n" + "=" * 60)
    print(f"Toys Generated:               {len(stats)}")
    print(f"Time Elapsed:                 {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Saved array to:               {out_file}")
    print("=" * 60)

if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--trigger', required=True)
    p.add_argument('--toys', type=int, default=1000)
    p.add_argument('--method', choices=["naive", "copula", "linear", "poisson_event", "exclusive_categories", "decorrelated_bootstrap"], required=True)
    p.add_argument('--gp-mean', choices=['5param', 'zero'], default='zero', help="Mean function used by the Gaussian Process")
    p.add_argument('--cms', type=float, default=13000.)
    p.add_argument('--jobid', type=str, default="local")
    p.add_argument('--min-len', type=float, default=0.15)
    p.add_argument('--fit', type=lambda x: (str(x).lower() == 'true'), default=True, help="Set to true to generate toys using a GP-smoothed frozen background")
    main(p.parse_args())
