#=============================================================
# Module for statistical analysis
#=============================================================
import numpy as np
from numpy import linalg as la
from scipy import stats
from collections import namedtuple

import matplotlib.pyplot as plt

#=============================================================
class StackArray:
    """
    Methods: \\
    add_array: add an array to stash \\
    stack: stack all arrays in the stash, output the stacked array, and reset stash \\
    unstack: split a stacked array using the indices in `stack_idx` and reshape it using the shapes stored in `shapes`
    """
    def __init__(self):
        self.stash = {}    # {name: array}
        self.shapes = {}    # {name: array_shapes}
        self.stack_idx = None    # [10, 25] for array sizes [10, 15, 20]
    
    def add_array(self, name, array):
        if name in self.stash:
            print(f"Overwriting the existing array ({name})!")
        self.stash[name] = np.asarray(array)

    def stack(self):
        if len(self.stash) == 0:
            print(f"Empty stash!")
            return None
        
        if self.stack_idx is None:    # No previous record of shapes
            sizes = []
            for name, arr in self.stash.items():
                sizes.append(arr.size)
                self.shapes[name] = arr.shape
            self.stack_idx = np.cumsum(sizes)[:-1]
        else:
            for name, arr in self.stash.items():    # Check the record of shapes
                if name not in self.shapes:
                    raise Exception(f"Input array ({name}) is not in the existing record!")
                elif self.shapes[name] != arr.shape:
                    raise Exception(f"Input array shape ({name}) does not match the existing record!")

        stacked = np.hstack([arr.ravel() for arr in self.stash.values()])
        # print(f'stacked={stacked.shape}, stack_idx={self.stack_idx}')
        self.stash = {}

        return stacked
    
    def unstack(self, stacked, nametuple_name='array', axis=0):
        """Unstack array using the stack indices."""
        if axis != 0:
            stacked = np.swapaxes(stacked, 0, axis)
        arrays = np.split(stacked, self.stack_idx, axis=0)
        arr_dict = {}
        for name, arr in zip(self.shapes.keys(), arrays):
            arr_dict[name] = np.squeeze(arr.reshape(self.shapes[name] + (-1,)))
        return namedtuple(nametuple_name, arr_dict.keys())(**arr_dict)
    
#=============================================================
def percentile_ci(boot_dist, alpha=0.05, axis=0):
    """Percentile bootstrap CI along a given axis."""
    lo = np.percentile(boot_dist, 100 * (alpha / 2), axis=axis)
    hi = np.percentile(boot_dist, 100 * (1 - alpha / 2), axis=axis)
    return lo, hi

def basic_ci(boot_dist, obs_stat, alpha=0.05, axis=0):
    """Basic bootstrap CI along a given axis."""
    lo = np.percentile(boot_dist, 100 * (alpha / 2), axis=axis)
    hi = np.percentile(boot_dist, 100 * (1 - alpha / 2), axis=axis)
    return 2 * obs_stat - hi, 2 * obs_stat - lo

def bca_ci(boot_dist, obs_stat, x, stat_func=None, axis=0, alpha=0.05):
    """
    Bias-Corrected and Accelerated (BCa) bootstrap CI.

    Parameters
    ----------
    boot_dist : ndarray
        Bootstrap replicates of the statistic (shape: [B, ...]).
    obs_stat : ndarray
        Observed statistic on original data (same shape as statistic).
    x : ndarray
        Original data array (axis specifies resampling dimension).
    stat_func : callable or None
        Function computing the statistic along 'axis'. If None, uses np.mean.
    axis : int
        Axis along which bootstrap and jackknife resampling occur.
    alpha : float
        Confidence level tail (e.g., 0.05 for 95% CI).

    Returns
    -------
    lo, hi : arrays
        Lower and upper BCa confidence limits.
    meta : dict
        Dictionary with z0, acceleration, and adjusted alphas.
    """
    stat_func = stat_func or (lambda d, axis=None: np.mean(d, axis=axis))
    x = np.asarray(x)
    boot_dist = np.asarray(boot_dist)
    obs_stat = np.asarray(obs_stat)

    # Move axis of interest to first position for consistent indexing
    x = np.moveaxis(x, axis, 0)
    n = x.shape[0]

    # --- 1) Bias correction z0 ---
    prop_less = np.mean(boot_dist < obs_stat, axis=0)
    prop_less = np.clip(prop_less, 1e-10, 1 - 1e-10)
    z0 = stats.norm.ppf(prop_less)

    # --- 2) Jackknife for acceleration ---
    jack_stats = np.empty((n,) + obs_stat.shape)
    for i in range(n):
        jack_sample = np.delete(x, i, axis=0)
        jack_stats[i] = stat_func(jack_sample, axis=0)
    jack_bar = np.mean(jack_stats, axis=0)
    num = np.sum((jack_bar - jack_stats) ** 3, axis=0)
    den = 6.0 * (np.sum((jack_bar - jack_stats) ** 2, axis=0) ** 1.5)
    a = np.divide(num, den, out=np.zeros_like(num), where=den != 0)

    # --- 3) Adjusted quantiles ---
    z_alpha = [stats.norm.ppf(alpha / 2), stats.norm.ppf(1 - alpha / 2)]
    adj_lo = stats.norm.cdf(z0 + (z0 + z_alpha[0]) / (1 - a * (z0 + z_alpha[0])))
    adj_hi = stats.norm.cdf(z0 + (z0 + z_alpha[1]) / (1 - a * (z0 + z_alpha[1])))

    # --- 4) Apply adjusted quantiles to bootstrap distribution ---
    lo = np.empty_like(obs_stat, dtype=float)
    hi = np.empty_like(obs_stat, dtype=float)

    it = np.nditer(obs_stat, flags=['multi_index'])
    while not it.finished:
        idx = it.multi_index
        lo[idx] = np.percentile(boot_dist[(slice(None),) + idx], 100 * adj_lo[idx])
        hi[idx] = np.percentile(boot_dist[(slice(None),) + idx], 100 * adj_hi[idx])
        it.iternext()

    meta = {"z0": z0, "a": a, "adj_alpha": (adj_lo, adj_hi)}
    return lo, hi, meta

def bootstrap(data, statistic, n_resamples=1000, confidence_level=0.95, method='BCa', random_state=42):
    """
    Perform bootstrap resampling to estimate the confidence interval of a statistic. This is similar to `scipy.stats.bootstrap` but allows more flexible statistic functions.
    Parameters: 
    data: shape(n_samples, n_features)
    method: 'percentile', 'basic', 'BCa'
        Resampling along the first dimension.
    """
    distribution = []

    rng = np.random.default_rng(random_state)
    for _ in range(n_resamples):
        idx = rng.choice(data.shape[0], size=data.shape[0], replace=True)
        sample = data[idx]
        # print(idx.shape, sample.shape)
        distribution.append(statistic(sample, axis=0))

    distribution = np.array(distribution)
    if method == 'percentile':
        confidence_interval = percentile_ci(distribution, alpha=1-confidence_level)
    elif method == 'basic':
        obs_stat = statistic(data, axis=0)
        confidence_interval = basic_ci(distribution, obs_stat, alpha=1-confidence_level)
    elif method == 'BCa':
        obs_stat = statistic(data, axis=0)
        confidence_interval = bca_ci(distribution, obs_stat, data, stat_func=statistic, axis=0, alpha=1-confidence_level)[:2]
    else:
        raise ValueError(f"Unknown method: {method}")
    standard_error = np.std(distribution, axis=0)
    # print(f"{confidence_level*100}% CI: ({confidence_interval})")
    # print(f"Standard error: {standard_error}")

    Result = namedtuple('Result', ['confidence_interval', 'bootstrap_distribution', 'standard_error'])
    return Result(confidence_interval, distribution, standard_error)

def bootstrap_model(data, boot_model, statistic, n_resamples=1000, confidence_level=0.95, method='BCa', random_state=42):
    """
    Perform bootstrap resampling to estimate the confidence interval of a statistic. This uses the input class `boot_model` in the general form of \\
        model= boot_model(data)
        for idx in range(n_resamples):
            sample = model.get_sample(random_state+idx)
    """
    distribution = []

    model = boot_model(data)
    for idx in range(n_resamples):
        sample = model.get_sample(random_state+idx*37)
        # print(idx.shape, sample.shape)
        distribution.append(statistic(sample, axis=0))

    distribution = np.array(distribution)
    if method == 'percentile':
        confidence_interval = percentile_ci(distribution, alpha=1-confidence_level)
    elif method == 'basic':
        obs_stat = statistic(data, axis=0)
        confidence_interval = basic_ci(distribution, obs_stat, alpha=1-confidence_level)
    elif method == 'BCa':
        obs_stat = statistic(data, axis=0)
        confidence_interval = bca_ci(distribution, obs_stat, data, stat_func=statistic, axis=0, alpha=1-confidence_level)[:2]
    else:
        raise ValueError(f"Unknown method: {method}")
    standard_error = np.std(distribution, axis=0)
    # print(f"{confidence_level*100}% CI: ({confidence_interval})")
    # print(f"Standard error: {standard_error}")

    Result = namedtuple('Result', ['confidence_interval', 'bootstrap_distribution', 'standard_error'])
    return Result(confidence_interval, distribution, standard_error)

def ttest_mean(data, alpha=0.05):
    """
    Compute t-based confidence interval for the mean.
    Perform t-test along the first dimension.
    """
    n = len(data)
    mean = np.mean(data, axis=0)
    s = np.std(data, axis=0, ddof=1)
    
    # t critical value
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
    
    # Standard error of the mean
    se = s / np.sqrt(n)
    
    # Confidence interval for mean
    ci_lower = mean - t_crit * se
    ci_upper = mean + t_crit * se
    
    ci = np.array([ci_lower, ci_upper])
    
    Result = namedtuple('Result', ['mean', 'confidence_interval', 'standard_error'])
    
    return Result(mean, ci, se)

def chi2_sd(data, alpha=0.05):
    """
    Compute chi-square-based confidence interval for the std.
    Perform test along the first dimension.
    """
    n = len(data)
    s = np.std(data, axis=0, ddof=1)
    
    # Chi-square critical values
    chi2_lower = stats.chi2.ppf(alpha/2, df=n-1)
    chi2_upper = stats.chi2.ppf(1 - alpha/2, df=n-1)
    
    # Confidence interval for SD
    # CI = sqrt((n-1) * s^2 / chi2)
    ci_lower = np.sqrt((n - 1) * s**2 / chi2_upper)
    ci_upper = np.sqrt((n - 1) * s**2 / chi2_lower)
    
    ci = np.array([ci_lower, ci_upper])
    
    # Standard error of SD (approximate)
    se = s / np.sqrt(2 * (n - 1))
    
    Result = namedtuple('Result', ['standard_deviation', 'confidence_interval', 'standard_error'])

    return Result(s, ci, se)

def get_ci_of_mean(y):
    """ Compare different methods to compute CI for the mean of y"""
    from scipy.stats import bootstrap as bootstrap_sp
    
    def stat_func(y, axis):
        return np.mean(y, axis=axis)
    
    sample_size = y.shape[0]
    sample_mean = y.mean(axis=0)
    res = ttest_mean(y, alpha=0.05)
    ci_t = res.confidence_interval

    res = bootstrap_sp((y,), statistic=stat_func, n_resamples=1000, confidence_level=0.95, random_state=42)
    ci0 = res.confidence_interval
    # plt.hist(res.bootstrap_distribution, bins=50, density=True)

    res = bootstrap(y, statistic=stat_func, n_resamples=1000, confidence_level=0.95, random_state=42, method='percentile')
    ci1 = res.confidence_interval
    # plt.hist(res.bootstrap_distribution, bins=50, density=True)

    res = bootstrap(y, statistic=stat_func, n_resamples=1000, confidence_level=0.95, random_state=42, method='basic')
    ci2 = res.confidence_interval

    res = bootstrap(y, statistic=stat_func, n_resamples=1000, confidence_level=0.95, random_state=42, method='BCa')
    ci3 = res.confidence_interval

    print("========================================")
    print(f"Sample size = {sample_size}")
    print(f"Sample mean = {sample_mean}")

    print("\nParametric t-based 95% CI:")
    print(f"  low=({ci_t[0]}), high=({ci_t[1]})")

    print("\nBootstrap 95% CIs:")

    print(f"  Scipy:           low=({ci0[0]}), high=({ci0[1]})")
    print(f"  My (percentile): low=({ci1[0]}), high=({ci1[1]})")
    print(f"  My (basic):      low=({ci2[0]}), high=({ci2[1]})")
    print(f"  My (BCa):        low=({ci3[0]}), high=({ci3[1]})")

def get_ci_of_std(y):
    """ Compare different methods to compute CI for the std of y"""
    from scipy.stats import bootstrap as bootstrap_sp
    
    def stat_func(y, axis):
        return np.std(y, axis=axis, ddof=1)
    
    sample_size = y.shape[0]
    sample_std = y.std(axis=0, ddof=1)
    res = chi2_sd(y, alpha=0.05)
    ci_t = res.confidence_interval

    res = bootstrap_sp((y,), statistic=stat_func, n_resamples=1000, confidence_level=0.95, random_state=42)
    ci0 = res.confidence_interval
    # plt.hist(res.bootstrap_distribution, bins=50, density=True)

    res = bootstrap(y, statistic=stat_func, n_resamples=1000, confidence_level=0.95, random_state=42, method='percentile')
    ci1 = res.confidence_interval
    # plt.hist(res.bootstrap_distribution, bins=50, density=True)

    res = bootstrap(y, statistic=stat_func, n_resamples=1000, confidence_level=0.95, random_state=42, method='basic')
    ci2 = res.confidence_interval

    res = bootstrap(y, statistic=stat_func, n_resamples=1000, confidence_level=0.95, random_state=42, method='BCa')
    ci3 = res.confidence_interval

    print("========================================")
    print(f"Sample size = {sample_size}")
    print(f"Sample std = {sample_std}")

    print("\nParametric chi2-based 95% CI:")
    print(f"  low=({ci_t[0]}), high=({ci_t[1]})")

    print("\nBootstrap 95% CIs:")

    print(f"  Scipy:           low=({ci0[0]}), high=({ci0[1]})")
    print(f"  My (percentile): low=({ci1[0]}), high=({ci1[1]})")
    print(f"  My (basic):      low=({ci2[0]}), high=({ci2[1]})")
    print(f"  My (BCa):        low=({ci3[0]}), high=({ci3[1]})")

def stats_tests_normal(N = 40, mu = 1.12, sigma = 2.57, seed=24):
    rng = np.random.default_rng(seed)
    y = rng.normal(loc=mu, scale=sigma, size=(N,2))

    print("========================================")
    print(f"True mean = {mu}")
    print(f"True standard deviation = {sigma}")
    get_ci_of_mean(y)
    get_ci_of_std(y)

def stats_tests_rand_model():
    from LIM_AM_mod import rand_model2, Myla
    from scipy import linalg as sla

    Q = np.array([[0.3, 0], [0, 0.1]])    # Q should be symmetric
    B = np.array([[-0.05, 0], [0, -0.1]])
    # B = np.array([[-0.05, -0.08], [0, -0.1]])

    # Q = np.array([[0.3, 0, 0], [0, 0.08, 0], [0, 0, 0.1]])    # Q should be symmetric
    # B = np.array([[-0.05, 0, 0], [0, -0.04, 0], [0, 0, -0.1]])
    # B = np.array([[-0.05, 0, 0], [-0.07, -0.04, 0], [0, 0, -0.1]])

    # Q =  - B @ C0 - C0 @ B.T
    C0 = -sla.solve_continuous_lyapunov(B, Q)
    b, _, _ = Myla.eig(B)

    y3 = rand_model2(B, Q, len_day=64*150, dt=0.2)
    print("========================================")
    print(f'\ndata={y3.shape}')
    print(f'sqrt(C0)={np.sqrt(C0)}')
    
    # get_ci_of_mean(y3)
    get_ci_of_std(y3)

#=============================================================
# main function
#=============================================================
if __name__ == "__main__":

    # stats_tests_normal(N = 40, mu = 1.12, sigma = 2.57)

    stats_tests_rand_model()

    pass