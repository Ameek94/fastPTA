"""
run_upperlimit_mcmc.py
======================
Zero-anisotropy I+V injection/recovery with MCMC.

The injected background is isotropic in intensity,

    c^I_00 = 1 / sqrt(4 pi),   c^I_lm = 0 for l >= 1,

and has no circular polarisation,

    c^V_lm = 0 for all observable V modes.

The MCMC samples the shared power-law spectrum and the anisotropy coefficients
for intensity and circular polarisation:

    log_amplitude, tilt, c^I_{l>=1,m}, c^V_{l>=1,m}.

The intensity monopole is fixed because it is degenerate with the amplitude, and
the V monopole is excluded because it is PTA-blind. The script reports posterior
95% upper limits on C_l^I/C_0^I and C_l^V/C_0^I using the same convention as
Cl_conventions_and_priors.md and the Fisher upper-limit scripts.

Usage
-----
    python run_upperlimit_mcmc.py
    python run_upperlimit_mcmc.py --quick
    python run_upperlimit_mcmc.py --l-max 2 --n-pulsars 100
    python run_upperlimit_mcmc.py --dry-run
"""

import argparse
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp
import emcee

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import corner
import healpy as hp

sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
)

import fastPTA.utils as ut
from fastPTA.signals import get_signal_model
from fastPTA.get_tensors import get_tensors
from fastPTA.data.generate_data import generate_MCMC_data
from fastPTA.inference_tools.likelihoods import log_likelihood
from fastPTA.angular_decomposition import spherical_harmonics as sph

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

L_MAX = 1
N_PULSARS = 60
T_OBS_YRS = 16.03
N_FREQUENCIES = 20

SIGNAL_MODEL = get_signal_model("power_law")
LOG_AMPLITUDE = -7.1995
TILT = 2.0
MONOPOLE = 1.0 / np.sqrt(4.0 * np.pi)
C0_I = MONOPOLE ** 2
PRIOR = 5.0 / (4.0 * np.pi)
LIMIT_CL = 0.95

LOGA_MIN, LOGA_MAX = -8.5, -5.5
TILT_MIN, TILT_MAX = 1.0, 3.0

# Use the expectation value by default so this is a clean upper-limit run.
REALIZATION = False

N_WALKERS = 24
BURNIN_STEPS = 1000
MCMC_STEPS = 2000
I_MAX = 10
R_CONVERGENCE = 5e-2

NSIDE_PRIOR = 8
NSIDE_PLOT = 32

OUTDIR = os.path.join(HERE, "generated_data")
PLOTDIR = os.path.join(HERE, "plots")
CATALOG_PATH = os.path.join(
    HERE, "..", "pulsar_configurations", "tmp_upperlimit_mcmc_IV.txt"
)
NOISELESS_YAML = os.path.join(
    HERE, "..", "pulsar_configurations",
    "EPTAlike_pulsar_parameters_noiseless.yaml",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def lm_labels(l_max, prefix):
    labels = []
    for ell in range(1, l_max + 1):
        for m in range(-ell, ell + 1):
            if m < 0:
                labels.append(f"{prefix}_{ell}m{abs(m)}")
            elif m == 0:
                labels.append(f"{prefix}_{ell}0")
            else:
                labels.append(f"{prefix}_{ell}p{m}")
    return labels


def lm_plot_labels(l_max, stokes):
    labels = []
    for ell in range(1, l_max + 1):
        for m in range(-ell, ell + 1):
            labels.append(rf"$c^{{{stokes}}}_{{{ell},{m}}}$")
    return labels


def real_ylm_basis(nside, n_lm, l_max):
    """Real spherical-harmonic basis maps Y[:, k], k = 0 .. n_lm-1."""
    npix = hp.nside2npix(nside)
    Y = np.zeros((npix, n_lm))
    for k in range(n_lm):
        e = np.zeros(n_lm)
        e[k] = 1.0
        Y[:, k] = sph.get_map_from_real_clms(e, nside, l_max=l_max)
    return Y


def zero_injection_coefficients(l_max):
    """Isotropic I and zero V coefficients."""
    n_lm = sph.get_n_coefficients_real(l_max)
    c_I = np.zeros(n_lm)
    c_I[0] = MONOPOLE
    c_V = np.zeros(n_lm - 1)
    return c_I, c_V


def cl_samples_from_chain(samples, l_max):
    """Return posterior samples of C_l^I/C_0^I and C_l^V/C_0^I."""
    n_lm = sph.get_n_coefficients_real(l_max)
    n_lm_V = n_lm - 1
    c_I = np.column_stack(
        [np.full(len(samples), MONOPOLE), samples[:, 2:2 + n_lm_V]]
    )
    c_V = np.column_stack(
        [np.zeros(len(samples)), samples[:, 2 + n_lm_V:]]
    )
    Cl_I = sph.get_CL_from_real_clm(c_I.T)[1:].T / C0_I
    Cl_V = sph.get_CL_from_real_clm(c_V.T)[1:].T / C0_I
    return Cl_I, Cl_V


def finite_initial_walkers(rng, truths, logp_np, n_walkers, n_lm_V):
    """
    Draw small perturbations around the zero-anisotropy truth, shrinking the
    coefficient scatter until every walker has finite posterior probability.
    """
    spec_scale = np.array([0.08, 0.08])
    coeff_scale = 0.015
    ndim = len(truths)

    for _ in range(12):
        scale = np.concatenate([spec_scale, coeff_scale * np.ones(2 * n_lm_V)])
        initial = truths[None, :] + scale[None, :] * rng.standard_normal(
            (n_walkers, ndim)
        )
        vals = logp_np(initial)
        if np.all(np.isfinite(vals)):
            return initial
        coeff_scale *= 0.5
        spec_scale *= 0.8

    raise RuntimeError(
        "Could not find finite initial walkers. Try reducing --l-max or "
        "increasing the physical-prior resolution only after checking the setup."
    )


def padded_limits(values, frac=0.05):
    lo = float(np.min(values))
    hi = float(np.max(values))
    if not np.isfinite(lo) or not np.isfinite(hi):
        return -1.0, 1.0
    if lo == hi:
        pad = max(abs(lo), 1.0) * frac
        return lo - pad, hi + pad
    pad = (hi - lo) * frac
    return lo - pad, hi + pad


def main(args):
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(PLOTDIR, exist_ok=True)

    n_lm = sph.get_n_coefficients_real(args.l_max)
    n_lm_V = n_lm - 1
    c_I_inj, c_V_inj = zero_injection_coefficients(args.l_max)

    names = (
        ["log_amplitude", "tilt"]
        + lm_labels(args.l_max, "cI")
        + lm_labels(args.l_max, "cV")
    )
    labels = (
        [r"$\log_{10}A$", r"$\gamma$"]
        + lm_plot_labels(args.l_max, "I")
        + lm_plot_labels(args.l_max, "V")
    )
    truths = np.concatenate(
        [[LOG_AMPLITUDE, TILT], c_I_inj[1:], c_V_inj]
    )
    ndim = len(names)
    min_walkers = 2 * ndim
    if args.n_walkers < min_walkers:
        print(f"  Increasing n_walkers from {args.n_walkers} to {min_walkers} "
              "for the emcee ensemble size.")
        args.n_walkers = min_walkers
    ell = np.arange(1, args.l_max + 1)

    print("=" * 66)
    print("  I+V MCMC upper limits — ZERO anisotropy injection")
    print("=" * 66)
    print(f"  N_pulsars={args.n_pulsars}  T_obs={T_OBS_YRS}yr  "
          f"n_freq={args.n_frequencies}  l_max={args.l_max}  "
          f"realization={REALIZATION}")
    print(f"  sampled parameters={ndim}  injected C_l^I/C_0^I=0, "
          f"C_l^V/C_0^I=0 for l>=1")

    Yp = real_ylm_basis(NSIDE_PRIOR, n_lm, args.l_max)
    I_inj = Yp @ c_I_inj
    V_inj = Yp @ np.concatenate([[0.0], c_V_inj])
    print(f"  min I(n)={I_inj.min():.4e}   max |V(n)|={np.max(np.abs(V_inj)):.4e}")

    if args.dry_run:
        print("\n  DRY RUN — setup only.")
        return

    # ----- ingredients -----
    frequency = (1.0 + jnp.arange(args.n_frequencies)) / (T_OBS_YRS * ut.yr)
    signal_std = np.sqrt(
        np.asarray(SIGNAL_MODEL.template(frequency, [LOG_AMPLITUDE, TILT]))
    )

    noiseless = ut.load_yaml(NOISELESS_YAML)
    strain_omega, response_IJ_lm, _, _, response_IJ_V_lm = get_tensors(
        frequency,
        path_to_pulsar_catalog=CATALOG_PATH,
        regenerate_catalog=True,
        anisotropies=True,
        circ_pol=True,
        l_max=args.l_max,
        n_pulsars=args.n_pulsars,
        save_catalog=True,
        use_ng_positions=False,
        **noiseless,
    )
    response_IJ_lm = np.asarray(response_IJ_lm)
    response_IJ_V_lm = np.asarray(response_IJ_V_lm)
    strain_omega = np.asarray(strain_omega)

    # ----- zero-anisotropy data, including a zero V block -----
    R_resp_inj = np.einsum("v,vfij->fij", c_I_inj, response_IJ_lm)
    A_resp_inj = np.einsum("v,vfij->fij", c_V_inj, response_IJ_V_lm)
    _, data, *_ = generate_MCMC_data(
        REALIZATION, frequency, signal_std, strain_omega,
        R_resp_inj, None, None,
        response_IJ_V=A_resp_inj, signal_std_V=signal_std,
        save_MCMC_data=False,
    )

    # ----- JAX posterior -----
    Yp_j = jnp.asarray(Yp)
    resp_I_j = jnp.asarray(response_IJ_lm)
    resp_V_j = jnp.asarray(response_IJ_V_lm)
    data_j = jnp.asarray(data)
    strain_j = jnp.asarray(strain_omega)

    def log_posterior_jax(theta):
        logA, tilt = theta[0], theta[1]
        cI = theta[2:2 + n_lm_V]
        cV = theta[2 + n_lm_V:]
        c_I_full = jnp.concatenate([jnp.array([MONOPOLE]), cI])
        c_V_full = jnp.concatenate([jnp.array([0.0]), cV])

        in_box = (
            (logA > LOGA_MIN) & (logA < LOGA_MAX)
            & (tilt > TILT_MIN) & (tilt < TILT_MAX)
            & (jnp.max(jnp.abs(cI)) < PRIOR)
            & (jnp.max(jnp.abs(cV)) < PRIOR)
        )
        I_map = Yp_j @ c_I_full
        V_map = Yp_j @ c_V_full
        ratio = jnp.max(jnp.abs(V_map) / jnp.where(I_map > 0, I_map, 1.0))
        phys = (jnp.min(I_map) > 0.0) & (ratio < 1.0)
        ok = in_box & phys

        R_resp = jnp.einsum("v,vfij->fij", c_I_full, resp_I_j)
        A_resp = jnp.einsum("v,vfij->fij", cV, resp_V_j)
        sv = SIGNAL_MODEL.template(frequency, jnp.array([logA, tilt]))
        ll = log_likelihood(
            data_j, sv, R_resp, strain_j,
            response_IJ_V=A_resp, signal_value_V=sv,
        )
        return jnp.where(ok, ll, -jnp.inf)

    logp_jit = jax.jit(log_posterior_jax)
    lp0 = float(logp_jit(jnp.asarray(truths)))
    print(f"\n  [JAX posterior] jit logp(truth) = {lp0:.3f}")

    logp_batched = jax.jit(jax.vmap(log_posterior_jax))

    def logp_np(thetas):
        return np.asarray(logp_batched(jnp.asarray(thetas)))

    rng = np.random.default_rng(args.seed)
    initial = finite_initial_walkers(rng, truths, logp_np, args.n_walkers, n_lm_V)

    print(f"\n  Running vectorised emcee: {args.n_walkers} walkers, "
          f"{ndim} parameters, {args.burnin_steps} burn-in + up to "
          f"{args.i_max}x{args.mcmc_steps} steps "
          f"(until |R-1| < {R_CONVERGENCE:g}) ...")
    t0 = time.time()
    sampler = emcee.EnsembleSampler(
        args.n_walkers, ndim, logp_np, vectorize=True
    )
    state = sampler.run_mcmc(initial, args.burnin_steps, progress=not args.quiet)
    sampler.reset()

    R_minus_1 = np.inf
    R_array = np.full(ndim, np.nan)
    for i in range(args.i_max):
        state = sampler.run_mcmc(state, args.mcmc_steps, progress=not args.quiet)
        R_array = ut.get_R(sampler.get_chain())
        R = np.sqrt(np.mean(R_array ** 2))
        R_minus_1 = np.abs(R - 1.0)
        print(f"  iteration {i + 1}/{args.i_max}: R-1 = {R_minus_1:.4f}  "
              f"(max per-parameter |R-1| = {np.max(np.abs(R_array - 1.0)):.4f})")
        if R_minus_1 < R_CONVERGENCE:
            print(f"  converged: R-1 = {R_minus_1:.4f} < {R_CONVERGENCE:g}")
            break
    else:
        print(f"  WARNING: reached i_max={args.i_max} without R-1 < "
              f"{R_CONVERGENCE:g} (final R-1 = {R_minus_1:.4f})")

    samples = sampler.get_chain(flat=True)
    pdfs = sampler.get_log_prob(flat=True)
    wall = time.time() - t0
    print(f"  sampling wall time: {wall:.1f}s   "
          f"mean acceptance: {np.mean(sampler.acceptance_fraction):.2f}   "
          f"final R-1 = {R_minus_1:.4f}")

    # ----- summaries and upper limits -----
    med = np.median(samples, axis=0)
    lo = np.quantile(samples, 0.16, axis=0)
    hi = np.quantile(samples, 0.84, axis=0)
    print("\n  Recovered (median [16,84]%) vs injected:")
    for i, name in enumerate(names):
        print(f"    {name:16s} = {med[i]:+.4f} [{lo[i]:+.4f}, {hi[i]:+.4f}]"
              f"   (injected {truths[i]:+.4f})")

    Cl_I_samples, Cl_V_samples = cl_samples_from_chain(samples, args.l_max)
    Cl_I_ul = np.quantile(Cl_I_samples, LIMIT_CL, axis=0)
    Cl_V_ul = np.quantile(Cl_V_samples, LIMIT_CL, axis=0)

    print(f"\n  {int(100 * LIMIT_CL)}% upper limits from MCMC "
          "(paper convention, C_l / C_0^I):")
    print(f"    {'l':>3}  {'C_l^I/C_0^I':>14}  {'C_l^V/C_0^I':>14}")
    for li, l in enumerate(ell):
        print(f"    {l:>3}  {Cl_I_ul[li]:>14.4e}  {Cl_V_ul[li]:>14.4e}")

    tag = f"upperlimit_mcmc_IV_lmax{args.l_max}_{args.n_pulsars}Np_{args.n_frequencies}Nf"
    chain_path = os.path.join(OUTDIR, f"{tag}_chains.npz")
    np.savez(
        chain_path,
        samples=samples,
        pdfs=pdfs,
        truths=truths,
        names=names,
        ell=ell,
        Cl_I_samples=Cl_I_samples,
        Cl_V_samples=Cl_V_samples,
        Cl_I_ul=Cl_I_ul,
        Cl_V_ul=Cl_V_ul,
        R_minus_1=R_minus_1,
        R_array=R_array,
        wall_time=wall,
        n_pulsars=args.n_pulsars,
        n_frequencies=args.n_frequencies,
        l_max=args.l_max,
    )
    print(f"\n  Saved chains and C_l samples to {chain_path}")

    # ----- corner plot -----
    if not args.no_corner:
        fig = corner.corner(
            samples, labels=labels, truths=truths, truth_color="C3",
            show_titles=True, title_fmt=".3f",
            title_kwargs={"fontsize": 8}, label_kwargs={"fontsize": 10},
        )
        fig.suptitle(
            f"zero-anisotropy I+V recovery ({args.n_pulsars} pulsars)",
            fontsize=13,
        )
        corner_path = os.path.join(PLOTDIR, f"{tag}_corner.png")
        fig.savefig(corner_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        print(f"  Corner plot saved to {corner_path}")

    # ----- C_l upper-limit plot -----
    fig, ax = plt.subplots(figsize=(6.2, 4.2), dpi=150)
    ax.plot(ell, Cl_I_ul, marker="o", color="tab:blue",
            label=r"$C_\ell^I/C_0^I$")
    ax.plot(ell, Cl_V_ul, marker="s", color="tab:red",
            label=r"$C_\ell^V/C_0^I$")
    ax.set_yscale("log")
    ax.set_xticks(ell)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"95% upper limit")
    ax.grid(alpha=0.3)
    ax.legend()
    ax.set_title("Zero-anisotropy I+V MCMC")
    fig.tight_layout()
    cl_path = os.path.join(PLOTDIR, f"{tag}_Cl_upperlimits.png")
    fig.savefig(cl_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  C_l upper-limit plot saved to {cl_path}")

    # ----- injected vs reconstructed sky maps -----
    c_I_med = np.concatenate([[MONOPOLE], med[2:2 + n_lm_V]])
    c_V_med = np.concatenate([[0.0], med[2 + n_lm_V:]])
    I_rec = sph.get_map_from_real_clms(c_I_med, NSIDE_PLOT, l_max=args.l_max)
    V_rec = sph.get_map_from_real_clms(c_V_med, NSIDE_PLOT, l_max=args.l_max)
    I_inj_plot = sph.get_map_from_real_clms(c_I_inj, NSIDE_PLOT, l_max=args.l_max)
    V_inj_plot = sph.get_map_from_real_clms(
        np.concatenate([[0.0], c_V_inj]), NSIDE_PLOT, l_max=args.l_max
    )

    Imin, Imax = padded_limits(np.concatenate([I_inj_plot, I_rec]))
    Vabs = max(np.abs(V_inj_plot).max(), np.abs(V_rec).max(), 1e-12)
    fig = plt.figure(figsize=(11, 7))
    hp.mollview(I_inj_plot, sub=(2, 2, 1), title="Injected $I(\\hat n)$",
                cmap="viridis", min=Imin, max=Imax)
    hp.mollview(I_rec, sub=(2, 2, 2), title="Posterior median $I(\\hat n)$",
                cmap="viridis", min=Imin, max=Imax)
    hp.mollview(V_inj_plot, sub=(2, 2, 3), title="Injected $V(\\hat n)$",
                cmap="RdBu_r", min=-Vabs, max=Vabs)
    hp.mollview(V_rec, sub=(2, 2, 4), title="Posterior median $V(\\hat n)$",
                cmap="RdBu_r", min=-Vabs, max=Vabs)
    map_path = os.path.join(PLOTDIR, f"{tag}_skymaps.png")
    fig.savefig(map_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Sky maps saved to {map_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pulsars", type=int, default=N_PULSARS)
    parser.add_argument("--n-frequencies", type=int, default=N_FREQUENCIES)
    parser.add_argument("--l-max", type=int, default=L_MAX)
    parser.add_argument("--n-walkers", type=int, default=N_WALKERS)
    parser.add_argument("--i-max", type=int, default=I_MAX)
    parser.add_argument("--burnin-steps", type=int, default=BURNIN_STEPS)
    parser.add_argument("--mcmc-steps", type=int, default=MCMC_STEPS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quick", action="store_true",
                        help="Small/fast smoke run.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the setup without running.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--no-corner", action="store_true",
                        help="Skip the corner plot.")
    args = parser.parse_args()

    if args.quick:
        args.n_pulsars = min(args.n_pulsars, 30)
        args.n_frequencies = 10
        args.l_max = min(args.l_max, 1)
        args.n_walkers = 20
        args.i_max = 2
        args.burnin_steps = 50
        args.mcmc_steps = 50

    main(args)
