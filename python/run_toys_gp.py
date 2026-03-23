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
    
    print(f"Initializing GP Baselines for {args.trigger.upper()}")
    
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

                counts_nom = FiveParam(args.cms, c, *[float(p) for p in d_nom['parameters']]) * widths

                if np.sum(counts_nom) > 0:
                    bkg_expectations[m] = counts_nom
                    channel_info[m] = {'centers': c, 'bins': v_bins, 'widths': widths}
                    
                    bkg_density = counts_nom / widths
                    bkg_err_density = np.sqrt(np.maximum(counts_nom, 1.0)) / widths
                    
                    locked_kernels[m] = get_optimized_background_kernel(
                        c, bkg_density, bkg_err_density, bkg_density, args.min_len
                    )
                    bkg_prior_densities[m] = bkg_density
                    
        except Exception:
            continue

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

    stats = []
    fit_failures, attempts = 0, 0
    max_attempts = args.toys * 50 
    
    print(f"\nGenerating {args.toys} {args.method} toys using LOCKED GP...")
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
                toy_density = toy_counts / w
                toy_err_density = np.sqrt(np.maximum(toy_counts, 1.0)) / w
                
                gp_density, fit_ok = fit_gp_locked(c, toy_density, toy_err_density, bkg_prior_densities[m], locked_kernels[m])
                if not fit_ok: 
                    toy_successful = False; break
                
                active_bkg = gp_density * w
                toy_max_t = max(toy_max_t, fast_bumphunter_stat(toy_counts, active_bkg))
        
        elif args.method == "linear":
            jj_b = bkg_expectations['jj']
            jj_pseudo = np.random.poisson(jj_b)
            jj_centers = channel_info['jj']['centers']

            for m, b in bkg_expectations.items():
                if m == 'jj':
                    toy_counts = jj_pseudo
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

                    # 2. Bivariate Poisson math (Safe against analytic fit crossing)
                    # The shared expectation cannot physically exceed the inclusive jj expectation
                    lambda_shared = np.minimum(b * ov_frac, jj_b_aligned)

                    # Calculate strict transfer probability (Guaranteed between 0 and 1)
                    p_transfer = lambda_shared / np.maximum(jj_b_aligned, 1e-15)

                    # 3. Draw correlated events
                    ov_counts = np.random.binomial(jj_pseudo_int, p_transfer)

                    # 4. Draw independent events to PERFECTLY restore the target sub-channel mean (b)
                    ind_b = np.maximum(0, b - lambda_shared)
                    ind_counts = np.random.poisson(ind_b)

                    # Total channel toy is exact integer math with a guaranteed mean of 'b'
                    toy_counts = ov_counts + ind_counts

                # --- GP Fitting and BumpHunter Evaluation ---
                c, w = channel_info[m]['centers'], channel_info[m]['widths']
                toy_density = toy_counts / w
                toy_err_density = np.sqrt(np.maximum(toy_counts, 1.0)) / w

                gp_density, fit_ok = fit_gp_locked(c, toy_density, toy_err_density, bkg_prior_densities[m], locked_kernels[m])
                if not fit_ok:
                    toy_successful = False; break

                active_bkg = gp_density * w
                toy_max_t = max(toy_max_t, fast_bumphunter_stat(toy_counts, active_bkg))


        elif args.method == "copula":
            sampled = matrix[np.random.choice(len(matrix), size=np.random.poisson(n_mother_exp), replace=True)]
            
            for m, b in bkg_expectations.items():
                idx = col_names.index(f"M{m}")
                
                # 1. Determine EXACT target yield based on analytic fit
                expected_yield = np.sum(b)
                target_n = np.random.poisson(expected_yield)
                
                if target_n == 0:
                    toy_counts = np.zeros(len(b), dtype=int)
                else:
                    # 2. Extract valid correlated quantiles
                    v_correlated = sampled[sampled[:, idx] >= 0, idx]
                    k = len(v_correlated)
                    
                    # 3. Reconcile matrix quantiles with target yield
                    if k >= target_n:
                        U_final = np.random.choice(v_correlated, size=target_n, replace=False)
                    else:
                        independent_n = target_n - k
                        U_independent = np.random.uniform(0, 1, size=independent_n)
                        U_final = np.concatenate([v_correlated, U_independent])
                        
                    # 4. Safe uniform dither to break up bootstrap duplicates
                    U_final += np.random.uniform(-0.0002, 0.0002, size=target_n)
                    
                    # 5. Boundary Reflection to prevent high-mass tail pile-up
                    U_final = np.abs(U_final)
                    U_final = np.where(U_final >= 1.0, 1.99999 - U_final, U_final)
                    
                    # 6. Map to physical histogram
                    toy_counts = np.bincount(np.searchsorted(cdfs[m], U_final), minlength=len(b))
                
                # --- GP Fitting and BumpHunter Evaluation ---
                c, w = channel_info[m]['centers'], channel_info[m]['widths']
                toy_density = toy_counts / w
                toy_err_density = np.sqrt(np.maximum(toy_counts, 1.0)) / w
                
                gp_density, fit_ok = fit_gp_locked(c, toy_density, toy_err_density, bkg_prior_densities[m], locked_kernels[m])
                if not fit_ok: 
                    toy_successful = False; break
                
                active_bkg = gp_density * w
                toy_max_t = max(toy_max_t, fast_bumphunter_stat(toy_counts, active_bkg))
        
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
    print(f"Time Elapsed:                 {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    print(f"Saved array to:               {out_file}")
    print("=" * 60)

if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--trigger', required=True)
    p.add_argument('--toys', type=int, default=1000)
    p.add_argument('--method', choices=["naive", "copula", "linear"], required=True)
    p.add_argument('--cms', type=float, default=13600.)
    p.add_argument('--jobid', type=str, default="local")
    p.add_argument('--min-len', type=float, default=0.15)
    main(p.parse_args())
