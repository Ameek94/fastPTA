"""
run_CVL_vs_npulsars.py
======================
Compute CVL (noiseless) Fisher-matrix upper limits on C_l^I and C_l^V for a
range of pulsar-array sizes.

For each N_pulsars the script:
  1. Generates N_realizations random catalog realizations (NANOGrav sky positions,
     noiseless EPTA-like noise).
  2. Computes the full I+V Fisher matrix via the doubled-real formulation.
  3. Extracts the block-diagonal I and V pieces.
  4. Samples the Fisher-Gaussian posterior and computes 95% MC upper limits on
     C_l^I and C_l^V (same method as the notebook).
  5. Computes the analytical chi^2 thresholds for cross-checking.
  6. Saves per-N_pulsars .npz files in generated_data/ and a summary .npz.

Usage
-----
    python run_CVL_vs_npulsars.py [--dry-run]

Outputs (in generated_data/)
----------------------------
    CVL_Npsr{N}.npz          — Fisher blocks + MC limits for each N_pulsars
    CVL_summary.npz           — arrays indexed by N_pulsars for quick plotting
"""

import argparse
import os
import sys
import time

import numpy as np
import tqdm
from scipy.stats import chi2

sys.path.insert(0, os.path.abspath("../"))

import fastPTA.utils as ut
from fastPTA.signals import get_signal_model
from fastPTA.Fisher_code import compute_fisher
from fastPTA.angular_decomposition import spherical_harmonics as sph

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Array sizes to sweep
N_PULSARS_LIST = [50, 100, 200, 500]

# Number of random catalog realizations per array size
N_REALIZATIONS = 30

# Observation time [yr]
T_OBS_YRS = 16.03

# Number of frequency bins
N_FREQUENCIES = 30

# Maximum angular multipole
L_MAX = 6

# Confidence level for upper limits
LIMIT_CL = 0.95

# Posterior samples per realization (MC limits)
N_POINTS = int(1e4)

# Prior bound on c_lm coefficients (flat prior: |c_lm| <= PRIOR)
PRIOR = 5.0 / (4.0 * np.pi)

# Signal model and fiducial parameters
SIGNAL_MODEL = get_signal_model("power_law")
LOG_AMPLITUDE = -7.1995
TILT = 2.0
SIGNAL_PARAMETERS = np.array([LOG_AMPLITUDE, TILT])
SHAPE_PARAMS = len(SIGNAL_PARAMETERS)  # = 2

# Force regeneration even if saved files exist
REGENERATE = False

# Output directory
OUTDIR = "generated_data"

# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

n_lm = sph.get_n_coefficients_real(L_MAX)
n_lm_V = n_lm - 1  # monopole excluded

signal_lm = 1e-30 / np.sqrt(4 * np.pi) * np.ones(n_lm)
signal_lm[0] = 1.0 / np.sqrt(4 * np.pi)
signal_lm_V = np.zeros(n_lm_V)

means_I = np.concatenate((SIGNAL_PARAMETERS, signal_lm[1:]))
means_V = signal_lm_V

monopole_idx = SHAPE_PARAMS  # = 2
n_I_block = SHAPE_PARAMS + n_lm_V
n_V_block = n_lm_V

# ---------------------------------------------------------------------------
# Load noiseless pulsar configuration
# ---------------------------------------------------------------------------

EPTAlike_noiseless = ut.load_yaml(
    "../pulsar_configurations/EPTAlike_pulsar_parameters_noiseless.yaml"
)

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_Cl_V_limits(means_V, cov_V, n_points, limit_cl, prior, max_iter=100):
    """Sample V-mode posterior and return (95% CL upper limits, prior-constrained limits)."""
    data = np.random.multivariate_normal(
        means_V, cov_V, n_points, check_valid="ignore", tol=1e-4
    )
    data_prior = data[np.max(np.abs(data), axis=-1) <= prior]
    i_add = 0
    while len(data_prior) < n_points and i_add < max_iter:
        extra = np.random.multivariate_normal(
            means_V, cov_V, 10 * n_points, check_valid="ignore", tol=1e-4
        )
        data_prior = np.vstack(
            [data_prior, extra[np.max(np.abs(extra), axis=-1) <= prior]]
        )
        i_add += 1

    dummy = np.zeros((len(data), 1))
    Cl_V = sph.get_CL_from_real_clm(np.hstack([dummy, data]).T)[1:]

    if len(data_prior) == 0:
        Cl_V_prior = np.full(len(Cl_V), np.nan)
    else:
        dummy_p = np.zeros((len(data_prior), 1))
        Cl_V_prior = sph.get_CL_from_real_clm(
            np.hstack([dummy_p, data_prior]).T
        )[1:]
        Cl_V_prior = np.quantile(Cl_V_prior, limit_cl, axis=-1)

    return np.quantile(Cl_V, limit_cl, axis=-1), Cl_V_prior


def get_lm_slice(l):
    """Indices within a (l_max+1)^2 lm-vector for multipole l."""
    return slice(l**2, l**2 + 2 * l + 1)


def analytical_Cl_det(cov_block, l, offset=0, limit_cl=LIMIT_CL):
    """
    Fast analytical 95% C_l threshold from Fisher covariance diagonal using
    the chi^2_{2l+1} distribution.  Multiply by 4pi to match the notebook
    C_l / C_0 normalisation.
    """
    n_modes = 2 * l + 1
    idx = get_lm_slice(l)
    variances = np.diag(cov_block)[offset + idx.start : offset + idx.stop]
    sigma2_mean = np.mean(variances)
    return sigma2_mean * chi2.ppf(limit_cl, df=n_modes) / n_modes * 4 * np.pi


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_for_n_pulsars(n_pulsars, dry_run=False):
    """Compute and save Fisher + C_l limits for a given array size."""
    outfile = os.path.join(OUTDIR, f"CVL_Npsr{n_pulsars}.npz")

    if not REGENERATE and os.path.exists(outfile):
        print(f"  [N={n_pulsars}] Loading existing {outfile}")
        data = np.load(outfile)
        return data

    if dry_run:
        print(f"  [N={n_pulsars}] DRY RUN — would generate {outfile}")
        return None

    print(f"\n{'='*60}")
    print(f"  N_pulsars = {n_pulsars}")
    print(f"{'='*60}")

    # Each n_pulsars value uses its own catalog file so runs never overwrite
    # each other.  use_ng_positions=False draws positions uniformly on the sky
    # so the catalog size truly varies with n_pulsars (use_ng_positions=True
    # would cap the catalog at the 68 real NANOGrav pulsars for any N > 68).
    catalog_path = os.path.join(
        "../pulsar_configurations", f"tmp_CVL_{n_pulsars}.txt"
    )

    generate_catalog_kwargs = {
        "n_pulsars": n_pulsars,
        "save_catalog": True,
        "use_ng_positions": False,
        **EPTAlike_noiseless,
    }

    get_tensors_kwargs = {
        "path_to_pulsar_catalog": catalog_path,
        "add_curn": False,
        "regenerate_catalog": True,
        "anisotropies": True,
        "circ_pol": True,
        "l_max": L_MAX,
    }

    fisher_kwargs = {
        "T_obs_yrs": T_OBS_YRS,
        "n_frequencies": N_FREQUENCIES,
        "signal_model": SIGNAL_MODEL,
        "signal_parameters": SIGNAL_PARAMETERS,
        "signal_lm": signal_lm,
        "signal_lm_V": signal_lm_V,
    }

    # --- Fisher matrices ---
    fisher_I = np.zeros((N_REALIZATIONS, n_I_block, n_I_block))
    fisher_V = np.zeros((N_REALIZATIONS, n_V_block, n_V_block))

    t0 = time.time()
    for idx in tqdm.tqdm(range(N_REALIZATIONS), desc=f"Fisher N={n_pulsars}"):
        res = np.array(compute_fisher(
            **fisher_kwargs,
            get_tensors_kwargs=get_tensors_kwargs,
            generate_catalog_kwargs=generate_catalog_kwargs,
        )[-1])
        res_red = np.delete(np.delete(res, monopole_idx, axis=0), monopole_idx, axis=1)
        fisher_I[idx] = res_red[:n_I_block, :n_I_block]
        fisher_V[idx] = res_red[n_I_block:, n_I_block:]

    print(f"  Fisher done in {time.time() - t0:.1f}s")

    # --- Covariances ---
    cov_I = ut.compute_inverse(fisher_I)
    cov_V = ut.compute_inverse(fisher_V)

    # --- MC C_l^I limits ---
    t0 = time.time()
    res_I = []
    for j in tqdm.tqdm(range(N_REALIZATIONS), desc=f"Cl^I MC N={n_pulsars}"):
        res_I.append(sph.get_Cl_limits(
            means_I, cov_I[j], SHAPE_PARAMS,
            n_points=N_POINTS, limit_cl=LIMIT_CL,
            max_iter=100, prior=PRIOR,
        ))
    res_I = np.array(res_I)
    Cl_I     = res_I[:, 0] * 4 * np.pi
    Cl_I_prior = res_I[:, 1] * 4 * np.pi
    print(f"  Cl^I MC done in {time.time() - t0:.1f}s")

    # --- MC C_l^V limits ---
    t0 = time.time()
    Cl_V_list, Cl_V_prior_list = [], []
    for j in tqdm.tqdm(range(N_REALIZATIONS), desc=f"Cl^V MC N={n_pulsars}"):
        cl_v, cl_v_prior = get_Cl_V_limits(
            means_V, cov_V[j], N_POINTS, LIMIT_CL, prior=PRIOR
        )
        Cl_V_list.append(cl_v * 4 * np.pi)
        Cl_V_prior_list.append(cl_v_prior * 4 * np.pi)
    Cl_V       = np.array(Cl_V_list)
    Cl_V_prior = np.array(Cl_V_prior_list)
    print(f"  Cl^V MC done in {time.time() - t0:.1f}s")

    # --- Analytical chi^2 thresholds ---
    ell = np.arange(1, L_MAX + 1)
    Cl_I_ana = np.array([
        [analytical_Cl_det(cov_I[j], l, offset=SHAPE_PARAMS) for l in range(1, L_MAX + 1)]
        for j in range(N_REALIZATIONS)
    ])
    Cl_V_ana = np.array([
        [analytical_Cl_det(cov_V[j], l, offset=0) for l in range(1, L_MAX + 1)]
        for j in range(N_REALIZATIONS)
    ])

    # --- Save ---
    np.savez(
        outfile,
        fisher_I=fisher_I,
        fisher_V=fisher_V,
        Cl_I=Cl_I,
        Cl_I_prior=Cl_I_prior,
        Cl_V=Cl_V,
        Cl_V_prior=Cl_V_prior,
        Cl_I_ana=Cl_I_ana,
        Cl_V_ana=Cl_V_ana,
        ell=ell,
        n_pulsars=n_pulsars,
    )
    print(f"  Saved {outfile}")
    return np.load(outfile)


def build_summary(results_by_N):
    """Aggregate per-N_pulsars results into a summary file."""
    ell = np.arange(1, L_MAX + 1)
    n_arr = np.array(sorted(results_by_N.keys()))

    mean_Cl_I     = np.array([np.mean(results_by_N[n]["Cl_I"],     axis=0) for n in n_arr])
    mean_Cl_V     = np.array([np.mean(results_by_N[n]["Cl_V"],     axis=0) for n in n_arr])
    mean_Cl_I_ana = np.array([np.mean(results_by_N[n]["Cl_I_ana"], axis=0) for n in n_arr])
    mean_Cl_V_ana = np.array([np.mean(results_by_N[n]["Cl_V_ana"], axis=0) for n in n_arr])

    outfile = os.path.join(OUTDIR, "CVL_summary.npz")
    np.savez(
        outfile,
        n_pulsars=n_arr,
        ell=ell,
        mean_Cl_I=mean_Cl_I,
        mean_Cl_V=mean_Cl_V,
        mean_Cl_I_ana=mean_Cl_I_ana,
        mean_Cl_V_ana=mean_Cl_V_ana,
    )
    print(f"\nSummary saved to {outfile}")
    print(f"  n_pulsars: {n_arr}")
    print(f"  mean C_l^I (l=1..{L_MAX}):")
    for n, row in zip(n_arr, mean_Cl_I):
        print(f"    N={n:4d}: {row}")
    print(f"  mean C_l^V (l=1..{L_MAX}):")
    for n, row in zip(n_arr, mean_Cl_V):
        print(f"    N={n:4d}: {row}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without running any computations."
    )
    parser.add_argument(
        "--n-pulsars", nargs="+", type=int, default=N_PULSARS_LIST,
        metavar="N",
        help=f"List of pulsar counts to sweep (default: {N_PULSARS_LIST})."
    )
    parser.add_argument(
        "--n-realizations", type=int, default=N_REALIZATIONS,
        help=f"Number of catalog realizations per array size (default: {N_REALIZATIONS})."
    )
    parser.add_argument(
        "--regenerate", action="store_true",
        help="Regenerate even if output files already exist."
    )
    args = parser.parse_args()

    if args.regenerate:
        REGENERATE = True

    os.makedirs(OUTDIR, exist_ok=True)

    print(f"CVL sweep: N_pulsars = {args.n_pulsars}")
    print(f"           N_realizations = {args.n_realizations}")
    print(f"           l_max = {L_MAX},  T_obs = {T_OBS_YRS} yr")

    results_by_N = {}
    for n_pulsars in args.n_pulsars:
        result = run_for_n_pulsars(n_pulsars, dry_run=args.dry_run)
        if result is not None:
            results_by_N[n_pulsars] = result

    if results_by_N and not args.dry_run:
        build_summary(results_by_N)
