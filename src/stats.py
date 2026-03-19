import numpy as np
from scipy.stats import poisson, norm
from scipy.special import gammainc

# =================================================================
# 1. Fast Vectorized BumpHunter Engine
# =================================================================
def fast_bumphunter_stat(data_hist, bkg_hist, max_width=30):
    """Vectorized sliding window BumpHunter. Returns max local test stat t = -ln(p)."""
    # 1. Replicate pyBumpHunter's array cropping (first to last non-zero bin)
    non_zero_idx = np.where(bkg_hist > 0)[0]
    if len(non_zero_idx) == 0:
        return 0.0

    h_inf, h_sup = non_zero_idx[0], non_zero_idx[-1] + 1
    d_crop = data_hist[h_inf:h_sup]
    b_crop = bkg_hist[h_inf:h_sup]

    max_t = 0.0
    for w in range(2, max_width + 1):
        k = np.ones(w)
        D = np.convolve(d_crop, k, mode='valid')
        B = np.convolve(b_crop, k, mode='valid')

        # 2. Only look for excesses where background > 0
        mask = (D > B) & (B > 0)
        if not np.any(mask):
            continue

        # 3. Use continuous incomplete gamma to perfectly match pyBH's fractional handling
        p = np.clip(gammainc(D[mask], B[mask]), 1e-300, 1.0)
        max_t = max(max_t, np.max(-np.log(p)))

    return max_t


# =================================================================
# 2. Pseudo-Experiment Generation
# =================================================================
def fast_bumphunter_pseudoexperiments(bkg_hist, num_toys=1000, max_width=30):
    """
    Runs standard BumpHunter pseudo-experiments (without background refitting).
    Throws Poisson toys around the expected background and calculates the max 
    test statistic for each toy. 
    
    Returns: numpy array of toy test statistics.
    """
    toy_stats = np.zeros(num_toys)
    
    # Pre-calculate the non-zero indices once to save time in the loop
    non_zero_idx = np.where(bkg_hist > 0)[0]
    if len(non_zero_idx) == 0:
        return toy_stats

    for i in range(num_toys):
        # Generate a Poisson toy from the background expectation
        toy_data = np.random.poisson(bkg_hist)
        
        # Calculate the test statistic for this toy
        t = fast_bumphunter_stat(toy_data, bkg_hist, max_width=max_width)
        toy_stats[i] = t
        
    return toy_stats


# =================================================================
# 3. Global Significance & p-value Calculators
# =================================================================
def calculate_global_pvalue(data_stat, toy_stats):
    """
    Calculates the Global p-value.
    p_global = (Number of toys with test stat >= data stat) / (Total Toys)
    """
    if len(toy_stats) == 0:
        return 1.0
        
    toy_stats_arr = np.array(toy_stats)
    n_greater_equal = np.sum(toy_stats_arr >= data_stat)
    
    # Calculate p-value (prevent absolute zero if data beats all toys)
    p_val = max(n_greater_equal / len(toy_stats), 1.0 / len(toy_stats))
    return p_val

def calculate_significance(p_value):
    """
    Converts a p-value into a Gaussian significance (Z-score).
    Uses the scipy.stats.norm inverse survival function.
    """
    if p_value >= 0.5:
        return 0.0
    
    # Cap at 8 sigma to prevent infinity errors on extremely small p-values
    if p_value <= 1.2e-15: 
        return 8.0 
        
    return norm.isf(p_value)

def evaluate_bumphunter_results(data_hist, bkg_hist, num_toys=1000, max_width=30):
    """
    Wrapper function: Runs the full standard BumpHunter routine on a single channel.
    Returns a dictionary of the local and global results.
    """
    # 1. Evaluate Data
    data_t = fast_bumphunter_stat(data_hist, bkg_hist, max_width=max_width)
    data_local_pval = np.exp(-data_t) if data_t > 0 else 1.0
    data_local_z = calculate_significance(data_local_pval)
    
    # 2. Evaluate Toys
    toy_stats = fast_bumphunter_pseudoexperiments(bkg_hist, num_toys=num_toys, max_width=max_width)
    
    # 3. Calculate Global Metrics
    global_pval = calculate_global_pvalue(data_t, toy_stats)
    global_z = calculate_significance(global_pval)
    
    return {
        'test_stat': data_t,
        'local_pval': data_local_pval,
        'local_z': data_local_z,
        'global_pval': global_pval,
        'global_z': global_z,
        'toy_stats': toy_stats
    }
