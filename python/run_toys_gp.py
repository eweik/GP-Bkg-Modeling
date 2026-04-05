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
from src.models import FiveParam, FiveParam_alt
from src.stats import fast_bumphunter_stat

def get_optimized_background_kernel(centers, density, density_err, parametric_density, min_len_scale_log):
    mask = density > 0
    X_log = np.log(centers[mask]).reshape(-1, 1)
    y_target = np.log(density[mask] / np.maximum(parametric_density[mask], 1e-15))
    y_err_target = density_err[mask] / density[mask]

    kernel = C(1.0, (1e-3, 1e2)) * RBF(length_scale=max(min_len_scale_log * 2.0, 0.5), length_scale_bounds=(min_len_scale_log, 5.0)) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 1e1))
    gp = GaussianProcessRegressor(kernel=kernel, alpha=y_err_target**2, n_restarts_optimizer=5, normalize_y=False)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X_log, y_target)
    return gp.kernel_

def fit_gp_locked(centers, toy_density, toy_err_density, prior_density, locked_kernel):
    mask = toy_density > 0
    if np.sum(mask) < 5:
        return prior_density, False

    X_log = np.log(centers[mask]).reshape(-1, 1)
    y_target = np.log(toy_density[mask] / np.maximum(prior_density[mask], 1e-15))
    y_err_target = toy_err_density[mask] / toy_density[mask]
    
    gp = GaussianProcessRegressor(kernel=locked_kernel, optimizer=None, alpha=y_err_target**2, normalize_y=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        gp.fit(X_log, y_target)
        
    X_full_log = np.log(centers).reshape(-1, 1)
    y_pred_target = gp.predict(X_full_log)
    return prior_density * np.exp(y_pred_target), True

def main(args):
    os.makedirs("results", exist_ok=True)
    base_dir = os.getcwd() if os.path.exists("data") and os.path.exists("fits") else repo_root

    mass_types = ["jj", "bb", "jb", "je", "jm", "jg", "be", "bm", "bg"]
    bkg_expectations, syst_envelopes, channel_info = {}, {}, {}
    locked_kernels, bkg_prior_densities = {}, {}
    overlap_map = TRIGGER_OVERLAPS.get(args.trigger.lower(), TRIGGER_OVERLAPS["default"])
    
    print(f"Initializing Baselines for {args.trigger.upper()}")
    
    # 1. Build Background and Lock GP Kernels (No ROOT files needed)
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

            counts_nom = FiveParam(args.cms, c, *[float(p) for p in d_nom['parameters']]) 

            if np.sum(counts_nom) > 0:
                bkg_expectations[m] = counts_nom
                channel_info[m] = {'centers': c, 'bins': v_bins, 'widths': widths}
                
                bkg_density = counts_nom / widths
                bkg_err_density = np.sqrt(np.maximum(counts_nom, 1.0)) / widths
                
                if args.fit:
                    locked_kernels[m] = get_optimized_background_kernel(
                        c, bkg_density, bkg_err_density, bkg_density, args.min_len
                    )
                bkg_prior_densities[m] = bkg_density
                
        except Exception:
            continue

    print(bkg_expectations["jj"])
    print(sum(bkg_expectations["jj"]))

    if not bkg_expectations:
        print("Error: No valid background fits found."); sys.exit(1)

    # 2. Copula Setup
    if args.method == "copula":
        copula_path = os.path.join(base_dir, "data", f"copula_{args.trigger}.npz")
        f = np.load(copula_path)
        matrix, col_names = f['copula'], list(f['columns'])
        cdfs = {m: np.cumsum(b) / np.sum(b) for m, b in bkg_expectations.items()}
        
        mother_key = 'jj' if 'jj' in bkg_expectations else list(bkg_expectations.keys())[0]
        n_mother_exp = np.sum(bkg_expectations[mother_key])
        channel_scales = {m: np.sum(b) / n_mother_exp for m, b in bkg_expectations.items()}

    elif args.method in ["poisson_event", "exclusive_categories"]:
        mass_path = os.path.join(base_dir, "data", f"masses_{args.trigger}.npz")
        f = np.load(mass_path)
        mass_matrix, col_names = f['masses'], list(f['columns'])
        N_events = len(mass_matrix)

        if args.method == "exclusive_categories":
            # Map events to orthogonal 9-bit categories
            valid_mask = mass_matrix > 0
            powers = 2 ** np.arange(len(col_names))
            event_patterns = valid_mask.dot(powers)
            unique_patterns = np.unique(event_patterns)
            pattern_indices = {p: np.where(event_patterns == p)[0] for p in unique_patterns}


    stats = []
    fit_failures, attempts = 0, 0
    max_attempts = args.toys * 50 
    
    fit_status_msg = "WITH LOCKED GP FITTING" if args.fit else "WITHOUT GP FITTING (Nominal Bkg)"
    print(f"\nGenerating {args.toys} {args.method} toys {fit_status_msg}...")
    start_time = time.time()
    
    # 3. Toy Loop
    while len(stats) < args.toys and attempts < max_attempts:
        attempts += 1
        toy_max_t = 0.0
        toy_successful = True
        
        completed = len(stats)
        if completed > 0 and completed % max(1, (args.toys // 20)) == 0:
            progress = int((completed / args.toys) * 100)
            sys.stdout.write(f"\rProgress: [{('=' * (progress//5)).ljust(20)}] {progress}% (Attempts: {attempts}) ")
            sys.stdout.flush()

        if args.method == "naive":
            for m, b in bkg_expectations.items():
                toy_counts = np.random.poisson(b)
                c, w = channel_info[m]['centers'], channel_info[m]['widths']
                
                if args.fit:
                    toy_density = toy_counts / w
                    toy_err_density = np.sqrt(np.maximum(toy_counts, 1.0)) / w
                    
                    gp_density, fit_ok = fit_gp_locked(c, toy_density, toy_err_density, bkg_prior_densities[m], locked_kernels[m])
                    if not fit_ok: 
                        toy_successful = False; break
                    active_bkg = gp_density * w
                else:
                    active_bkg = b
                
                toy_max_t = max(toy_max_t, fast_bumphunter_stat(toy_counts, active_bkg))
        
        elif args.method == "linear":
            jj_b = bkg_expectations.get('jj', None)
            if jj_b is not None:
                jj_pseudo = np.random.poisson(jj_b)
                jj_centers = channel_info['jj']['centers']

            for m, b in bkg_expectations.items():
                if m == 'jj':
                    toy_counts = jj_pseudo
                else:
                    if jj_b is None:
                        # Fallback if jj fit is entirely missing
                        toy_counts = np.random.poisson(b)
                    else:
                        ov_frac = overlap_map.get(m, 0.1)
                        m_centers = channel_info[m]['centers']

                        # 1. Exact Bin Alignment
                        jj_b_aligned = np.zeros(len(b))
                        jj_pseudo_int = np.zeros(len(b), dtype=int)

                        for i, mc in enumerate(m_centers):
                            dist = np.abs(jj_centers - mc)
                            min_idx = np.argmin(dist)
                            if dist[min_idx] < 1.0:
                                jj_b_aligned[i] = jj_b[min_idx]
                                jj_pseudo_int[i] = jj_pseudo[min_idx]

                        # 2. The Physical Cap
                        expected_overlap = np.minimum(b * ov_frac, jj_b_aligned)

                        # 3. Capped Linear Transfer 
                        safe_jj_b = np.maximum(jj_b_aligned, 1e-15)
                        fluctuated_overlap = expected_overlap * (jj_pseudo_int / safe_jj_b)

                        # 4. Fluctuate the independent remainder
                        expected_remaining = np.maximum(0.0, b - expected_overlap)
                        ind_counts = np.random.poisson(expected_remaining)

                        # 5. Total channel toy
                        toy_counts = fluctuated_overlap + ind_counts

                # --- GP Fitting and BumpHunter Evaluation ---
                c, w = channel_info[m]['centers'], channel_info[m]['widths']
                
                if args.fit:
                    toy_density = toy_counts / w
                    toy_err_density = np.sqrt(np.maximum(toy_counts, 1.0)) / w

                    gp_density, fit_ok = fit_gp_locked(c, toy_density, toy_err_density, bkg_prior_densities[m], locked_kernels[m])
                    if not fit_ok:
                        toy_successful = False; break
                    active_bkg = gp_density * w
                else:
                    active_bkg = b

                toy_max_t = max(toy_max_t, fast_bumphunter_stat(toy_counts, active_bkg))


        elif args.method == "copula":
            sampled = matrix[np.random.choice(len(matrix), size=np.random.poisson(n_mother_exp), replace=True)]
            
            for m, b in bkg_expectations.items():
                idx = col_names.index(f"M{m}")
                
                expected_yield = np.sum(b)
                target_n = np.random.poisson(expected_yield)
                
                if target_n == 0:
                    toy_counts = np.zeros(len(b), dtype=int)
                else:
                    v_correlated = sampled[sampled[:, idx] >= 0, idx]
                    k = len(v_correlated)
                    
                    if k >= target_n:
                        U_final = np.random.choice(v_correlated, size=target_n, replace=False)
                    else:
                        independent_n = target_n - k
                        U_independent = np.random.uniform(0, 1, size=independent_n)
                        U_final = np.concatenate([v_correlated, U_independent])
                        
                    U_final += np.random.uniform(-0.0002, 0.0002, size=target_n)
                    U_final = np.abs(U_final)
                    U_final = np.where(U_final >= 1.0, 1.99999 - U_final, U_final)
                    
                    toy_counts = np.bincount(np.searchsorted(cdfs[m], U_final), minlength=len(b))
                
                # --- GP Fitting and BumpHunter Evaluation ---
                c, w = channel_info[m]['centers'], channel_info[m]['widths']
                
                if args.fit:
                    toy_density = toy_counts / w
                    toy_err_density = np.sqrt(np.maximum(toy_counts, 1.0)) / w
                    
                    gp_density, fit_ok = fit_gp_locked(c, toy_density, toy_err_density, bkg_prior_densities[m], locked_kernels[m])
                    if not fit_ok: 
                        toy_successful = False; break
                    
                    active_bkg = gp_density * w
                else:
                    active_bkg = b

                toy_max_t = max(toy_max_t, fast_bumphunter_stat(toy_counts, active_bkg))
        
        elif args.method in ["poisson_event", "exclusive_categories"]:
            if args.method == "poisson_event":
                # Approach 1: Draw X ~ Poisson(N) globally
                N_draw = np.random.poisson(N_events)
                sampled_rows = mass_matrix[np.random.choice(N_events, size=N_draw, replace=True)]
                # event_weights = np.random.poisson(1.0, size=N_events)
            else:
                # Approach 2: Draw n_toy_c ~ Poisson(n_c) for each orthogonal 9-bit pattern
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

            # Reconstruct the binned spectra from the sampled physical events
            for m, b in bkg_expectations.items():
                idx = col_names.index(f"M{m}")
                masses = sampled_rows[:, idx]
                valid_masses = masses[masses > 0]  * args.cms
                toy_counts, _ = np.histogram(valid_masses, bins=channel_info[m]['bins'])
                # print(toy)
                # print(b)
                print(f"Mass: {m}")
                print(f"Sum of toys: {sum(toy_counts)}")
                print(f"Sum of background: {sum(b)}")

                if np.sum(toy_counts) < 50: continue

                if args.fit:
                    toy_density = toy_counts / w
                    toy_err_density = np.sqrt(np.maximum(toy_counts, 1.0)) / w
                    
                    gp_density, fit_ok = fit_gp_locked(c, toy_density, toy_err_density, bkg_prior_densities[m], locked_kernels[m])
                    if not fit_ok: 
                        toy_successful = False; break
                    
                    active_bkg = gp_density * w
                else:
                    # CRITICAL NORMALIZATION FIX: Scale frozen background to fluctuating toy integral
                    # active_bkg = b * (np.sum(toy) / np.sum(b))
                    active_bkg = b

                toy_max_t = max(toy_max_t, fast_bumphunter_stat(toy_counts, active_bkg))
            print(f"Max-test-statistic: {toy_max_t}")


        if toy_successful:
            stats.append(toy_max_t)
        else:
            fit_failures += 1

    sys.stdout.write(f"\rProgress: [{'=' * 20}] 100% \n")
    sys.stdout.flush()

    out_file = os.path.join("results", f"global_stat_GP_{args.trigger}_{args.method}_{args.jobid}.npy")
    np.save(out_file, stats)
    
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    
    print("\n" + "=" * 60)
    print(f"Toys Generated:               {len(stats)}")
    print(f"Fit Failures Skipped:         {fit_failures}")
    print(f"Time Elapsed:                 {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Saved array to:               {out_file}")
    print("=" * 60)

if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--trigger', required=True)
    p.add_argument('--toys', type=int, default=1000)
    p.add_argument('--method', choices=["naive", "copula", "linear", "poisson_event", "exclusive_categories"], required=True)
    p.add_argument('--cms', type=float, default=13600.)
    p.add_argument('--jobid', type=str, default="local")
    p.add_argument('--min-len', type=float, default=0.15)
    p.add_argument('--fit', type=lambda x: (str(x).lower() == 'true'), default=False, help="Set to true to use GP fitting, false to use nominal analytic background (default: true)")
    main(p.parse_args())
