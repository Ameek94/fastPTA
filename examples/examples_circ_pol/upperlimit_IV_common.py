"""
upperlimit_IV_common.py
=======================
Shared machinery for the I+V circular-polarisation Fisher upper-limit forecasts:

  * run_upperlimit_IV_isotropic.py  — zero I and V anisotropy injection
  * run_upperlimit_IV_Ianiso.py     — I anisotropy injected, zero V injection

Both produce noiseless (CVL) Fisher 95% upper limits on the angular power spectra,
reported in the paper convention (see Cl_conventions_and_priors.md):

  * real spherical-harmonic basis, c^I_00 = 1/sqrt(4 pi)  ->  C_0^I = 1/(4 pi)
  * C_l = (1/(2l+1)) sum_m c_lm^2          (sph.get_CL_from_real_clm)
  * V monopole excluded (PTA-blind); the V power is normalised by the INTENSITY
    monopole C_0^I, i.e. reported as C_l^V / C_0^I
  * flat box prior |c_lm| < 5/(4 pi); all limits returned as C_l / C_0^I, which is
    the bare C_l multiplied by 4 pi (since 1 / C_0^I = 4 pi)

The I and V Fisher blocks are extracted block-diagonally. This is exact for both
scenarios: at a zero-V injection the I-V cross Fisher vanishes (it is the trace of a
symmetric times an antisymmetric matrix), so I and V decouple. The V auto-block
still depends on the injected I anisotropy through the covariance C = R(c^I).
"""

import os
import sys
import time

import numpy as np
import tqdm
from scipy.stats import chi2

# Make the repository importable regardless of the working directory or install
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

import fastPTA.utils as ut
from fastPTA.signals import get_signal_model
from fastPTA.Fisher_code import compute_fisher
from fastPTA.angular_decomposition import spherical_harmonics as sph

# ---------------------------------------------------------------------------
# Shared configuration / conventions
# ---------------------------------------------------------------------------

T_OBS_YRS = 16.03
N_FREQUENCIES = 30
LIMIT_CL = 0.95
N_POINTS = int(1e4)
PRIOR = 5.0 / (4.0 * np.pi)          # flat box prior on |c_lm|
C0_I = 1.0 / (4.0 * np.pi)           # intensity monopole power  C_0^I
INV_C0_I = 4.0 * np.pi               # 1 / C_0^I  (multiply bare C_l by this)

SIGNAL_MODEL = get_signal_model("power_law")
LOG_AMPLITUDE = -7.1995
TILT = 2.0
SIGNAL_PARAMETERS = np.array([LOG_AMPLITUDE, TILT])
SHAPE_PARAMS = len(SIGNAL_PARAMETERS)  # = 2

PULSAR_CONFIG_DIR = os.path.join(HERE, "..", "pulsar_configurations")
NOISELESS_YAML = os.path.join(
    PULSAR_CONFIG_DIR, "EPTAlike_pulsar_parameters_noiseless.yaml"
)


# ---------------------------------------------------------------------------
# Injection helpers (paper convention)
# ---------------------------------------------------------------------------

def isotropic_signal_lm(l_max):
    """Real c^I_lm vector for an isotropic (monopole-only) intensity."""
    n_lm = sph.get_n_coefficients_real(l_max)
    signal_lm = np.zeros(n_lm)
    signal_lm[0] = 1.0 / np.sqrt(4.0 * np.pi)   # c^I_00 = 1/sqrt(4 pi)
    return signal_lm


def add_single_m_mode(signal_lm, l, m, Cl_over_C0, l_max):
    """
    Add a single real (l, m) intensity mode at target C_l/C_0^I = Cl_over_C0.

    For a single excited m, C_l = c_lm^2 / (2l+1), so
        c_lm = sqrt((2l+1) * Cl_over_C0) * c_00 .
    """
    signal_lm = np.array(signal_lm, dtype=float)
    c00 = signal_lm[0]
    idx = sph.get_n_coefficients_real(l - 1) + (m + l)   # real-lm ordering
    signal_lm[idx] = np.sqrt((2 * l + 1) * Cl_over_C0) * c00
    return signal_lm


# ---------------------------------------------------------------------------
# Fisher matrices
# ---------------------------------------------------------------------------

def fisher_blocks(n_pulsars, signal_lm, signal_lm_V, l_max, n_realizations,
                  catalog_tag, n_frequencies=N_FREQUENCIES,
                  t_obs_yrs=T_OBS_YRS):
    """
    Compute n_realizations noiseless I+V Fisher matrices and return the
    block-diagonal I and V pieces (monopole removed from the I block).

    Returns
    -------
    fisher_I : (n_realizations, SHAPE_PARAMS + n_lm_V, ...) array
    fisher_V : (n_realizations, n_lm_V, n_lm_V) array
    """
    n_lm = sph.get_n_coefficients_real(l_max)
    n_lm_V = n_lm - 1
    n_I_block = SHAPE_PARAMS + n_lm_V
    monopole_idx = SHAPE_PARAMS                       # index of c^I_00 in the full matrix

    EPTAlike_noiseless = ut.load_yaml(NOISELESS_YAML)
    catalog_path = os.path.join(
        PULSAR_CONFIG_DIR, f"tmp_{catalog_tag}_{n_pulsars}.txt"
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
        "l_max": l_max,
    }
    fisher_kwargs = {
        "T_obs_yrs": t_obs_yrs,
        "n_frequencies": n_frequencies,
        "signal_model": SIGNAL_MODEL,
        "signal_parameters": SIGNAL_PARAMETERS,
        "signal_lm": signal_lm,
        "signal_lm_V": signal_lm_V,
    }

    fisher_I = np.zeros((n_realizations, n_I_block, n_I_block))
    fisher_V = np.zeros((n_realizations, n_lm_V, n_lm_V))

    t0 = time.time()
    for idx in tqdm.tqdm(range(n_realizations), desc=f"Fisher N={n_pulsars}"):
        res = np.array(compute_fisher(
            **fisher_kwargs,
            get_tensors_kwargs=get_tensors_kwargs,
            generate_catalog_kwargs=generate_catalog_kwargs,
        )[-1])
        res_red = np.delete(
            np.delete(res, monopole_idx, axis=0), monopole_idx, axis=1
        )
        fisher_I[idx] = res_red[:n_I_block, :n_I_block]
        fisher_V[idx] = res_red[n_I_block:, n_I_block:]
    print(f"  Fisher done in {time.time() - t0:.1f}s")

    return fisher_I, fisher_V


# ---------------------------------------------------------------------------
# C_l upper limits (returned as C_l / C_0^I)
# ---------------------------------------------------------------------------

def cl_limits_I(means_I, cov_I, n_points=N_POINTS, limit_cl=LIMIT_CL,
                prior=PRIOR):
    """95% C_l^I / C_0^I upper limits per realization (with and without prior)."""
    out, out_prior = [], []
    for j in tqdm.tqdm(range(len(cov_I)), desc="Cl^I MC"):
        cl, cl_prior = sph.get_Cl_limits(
            means_I, cov_I[j], SHAPE_PARAMS,
            n_points=n_points, limit_cl=limit_cl, max_iter=100, prior=prior,
        )
        out.append(cl * INV_C0_I)
        out_prior.append(cl_prior * INV_C0_I)
    return np.array(out), np.array(out_prior)


def cl_limits_V(means_V, cov_V, n_points=N_POINTS, limit_cl=LIMIT_CL,
                prior=PRIOR, max_iter=100):
    """
    95% C_l^V / C_0^I upper limits per realization (with and without prior).

    The V coefficient vector has no monopole, so a dummy zero monopole slot is
    prepended before calling get_CL_from_real_clm (which expects an l=0 entry).
    """
    out, out_prior = [], []
    for j in tqdm.tqdm(range(len(cov_V)), desc="Cl^V MC"):
        data = np.random.multivariate_normal(
            means_V, cov_V[j], n_points, check_valid="ignore", tol=1e-4
        )
        data_prior = data[np.max(np.abs(data), axis=-1) <= prior]
        i_add = 0
        while len(data_prior) < n_points and i_add < max_iter:
            extra = np.random.multivariate_normal(
                means_V, cov_V[j], 10 * n_points, check_valid="ignore", tol=1e-4
            )
            data_prior = np.vstack(
                [data_prior, extra[np.max(np.abs(extra), axis=-1) <= prior]]
            )
            i_add += 1

        dummy = np.zeros((len(data), 1))
        Cl_V = sph.get_CL_from_real_clm(np.hstack([dummy, data]).T)[1:]
        cl = np.quantile(Cl_V, limit_cl, axis=-1) * INV_C0_I

        if len(data_prior) == 0:
            cl_prior = np.full(len(cl), np.nan)
        else:
            dummy_p = np.zeros((len(data_prior), 1))
            Cl_V_prior = sph.get_CL_from_real_clm(
                np.hstack([dummy_p, data_prior]).T
            )[1:]
            cl_prior = np.quantile(Cl_V_prior, limit_cl, axis=-1) * INV_C0_I

        out.append(cl)
        out_prior.append(cl_prior)
    return np.array(out), np.array(out_prior)


def cl_analytical(cov_block, l_max, offset, limit_cl=LIMIT_CL):
    """
    Analytical chi^2_{2l+1} 95% C_l / C_0^I thresholds from the Fisher
    covariance diagonal (fast cross-check of the MC limits).
    """
    out = np.zeros(l_max)
    for li, l in enumerate(range(1, l_max + 1)):
        idx = slice(l ** 2, l ** 2 + 2 * l + 1)
        var = np.diag(cov_block)[offset + idx.start: offset + idx.stop]
        out[li] = np.mean(var) * chi2.ppf(limit_cl, df=2 * l + 1) / (2 * l + 1) \
            * INV_C0_I
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_limits(ell, Cl_I, Cl_V, title, outpath, injected_I=None,
                ng15_dat=None):
    """
    Two-panel figure: 95% C_l/C_0^I upper limits vs l for I (left) and V (right),
    with the mean and [16, 84]% spread over realizations shown as a band.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def band(ax, Cl, color, label):
        mean = np.mean(Cl, axis=0)
        lo = np.quantile(Cl, 0.16, axis=0)
        hi = np.quantile(Cl, 0.84, axis=0)
        ax.fill_between(ell, lo, hi, color=color, alpha=0.25)
        ax.plot(ell, mean, color=color, marker="o", lw=1.5, label=label)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=150, sharex=True)

    band(axes[0], Cl_I, "tab:blue", "Fisher 95% UL")
    if injected_I is not None:
        axes[0].plot(ell, injected_I, "k--", marker="x", lw=1,
                     label="injected $C_\\ell^I/C_0^I$")
    if ng15_dat is not None and os.path.exists(ng15_dat):
        ng = np.loadtxt(ng15_dat)
        axes[0].plot(ng[:, 0], ng[:, 1], "g+", lw=0, ms=9,
                     label="NANOGrav 15-yr (noisy)")
    axes[0].set_ylabel(r"$C_\ell^I / C_0^I$  (95% upper limit)")
    axes[0].set_title("Intensity")

    band(axes[1], Cl_V, "tab:red", "Fisher 95% UL")
    axes[1].set_ylabel(r"$C_\ell^V / C_0^I$  (95% upper limit)")
    axes[1].set_title("Circular polarisation")

    for ax in axes:
        ax.set_yscale("log")
        ax.set_xlabel(r"$\ell$")
        ax.set_xticks(ell)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved to {outpath}")
