"""
run_injection_l1_IV.py
======================
Inject an ell = 1 (dipole) anisotropy in BOTH intensity (Stokes I) and circular
polarisation (Stokes V), generate the corresponding mock PTA data with the
complex-Hermitian covariance ``C = R + i A``, and recover the signal with an
MCMC that samples the *anisotropy coefficients* c^I_lm and c^V_lm together with
the (shared) power-law spectrum.

Following the fastPTA Fisher convention (examples_circ_pol/run_CVL_vs_npulsars.py)
the intensity and V-mode share a single power-law spectrum {log_amplitude, tilt};
the V amplitude is carried by the c^V_lm coefficients. The intensity monopole is
fixed to c^I_00 = 1/sqrt(4 pi) (degenerate with the amplitude) and the V monopole
is excluded (a PTA is blind to it). For l_max = 1 the sampled parameters are

    log_amplitude, tilt,
    c^I_{1,-1}, c^I_{1,0}, c^I_{1,1},
    c^V_{1,-1}, c^V_{1,0}, c^V_{1,1}.

Outputs (in generated_data/ and plots/):
  * corner plot of the (isotropic + anisotropic) parameters with injected truths
  * injected vs reconstructed I and V sky maps, side by side

Usage
-----
    python run_injection_l1_IV.py            # full run
    python run_injection_l1_IV.py --quick    # small/fast smoke run
    python run_injection_l1_IV.py --dry-run  # print the setup only
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

# Make the repository importable regardless of the working directory
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

L_MAX = 1                         # inject only the ell = 1 dipole

# Defaults chosen so a full run completes in a few minutes; larger N_PULSARS
# tightens the (intrinsically harder) V-mode constraint at extra cost.
N_PULSARS = 60
T_OBS_YRS = 16.03
N_FREQUENCIES = 20

# Shared power-law spectrum (I and V); SMBHB fiducial
SIGNAL_MODEL = get_signal_model("power_law")
LOG_AMPLITUDE = -7.1995
TILT = 2.0

# Injected ell = 1 mode specified by its angular power in the PAPER CONVENTION
# (see Cl_conventions_and_priors.md):
#   C_l = (1/(2l+1)) sum_m c_lm^2,   C_0 = |c_00|^2 = 1/(4 pi)
# (i.e. sph.get_CL_from_real_clm). A single m is excited, so the coefficient at a
# target C_l/C_0 = C_ELL is
#   c_lm = sqrt((2l+1) * C_ELL) * c_00 .
# The V dipole carries no monopole, so its power is normalised by the INTENSITY
# monopole C_0^I (there is no C_0^V): C_ELL is C_1^I/C_0^I for I and C_1^V/C_0^I
# for V. The I and V dipoles are on PERPENDICULAR axes (I along z, V along x) so
# that I>0 and |V|<=I hold everywhere.
C_ELL = 0.05                      # C_l / C_0^I  for the injected ell = 1 mode
MONOPOLE = 1.0 / np.sqrt(4.0 * np.pi)
COEFF = np.sqrt((2 * L_MAX + 1) * C_ELL)   # c_lm / c_00 for a single excited m
DIPOLE_I = COEFF * MONOPOLE        # intensity c^I_{1, 0}  (z-axis)
DIPOLE_V = COEFF * MONOPOLE        # V-mode    c^V_{1, 1}  (x-axis, perp)
PRIOR = 5.0 / (4.0 * np.pi)       # flat box prior on |c_lm|

# Spectrum priors
LOGA_MIN, LOGA_MAX = -8.5, -5.5
TILT_MIN, TILT_MAX = 1.0, 3.0

# Data realization vs expectation (face value)
REALIZATION = False

# MCMC settings
N_WALKERS = 24
BURNIN_STEPS = 1000
MCMC_STEPS = 2000
I_MAX = 10
R_CONVERGENCE = 5e-2

# Resolutions
NSIDE_PRIOR = 8                   # for the positivity / |V|<=I prior check
NSIDE_PLOT = 32                   # for the reconstructed sky maps

OUTDIR = os.path.join(HERE, "generated_data")
PLOTDIR = os.path.join(HERE, "plots")
CATALOG_PATH = os.path.join(
    HERE, "..", "pulsar_configurations", "tmp_injection_l1_IV.txt"
)
NOISELESS_YAML = os.path.join(
    HERE, "..", "pulsar_configurations",
    "EPTAlike_pulsar_parameters_noiseless.yaml",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def real_ylm_basis(nside, n_lm, l_max):
    """Real spherical-harmonic basis maps Y[:, k], k = 0 .. n_lm-1."""
    npix = hp.nside2npix(nside)
    Y = np.zeros((npix, n_lm))
    for k in range(n_lm):
        e = np.zeros(n_lm)
        e[k] = 1.0
        Y[:, k] = sph.get_map_from_real_clms(e, nside, l_max=l_max)
    return Y


def injected_coefficients():
    """Injected real coefficient vectors (ordered by (l, m))."""
    n_lm = sph.get_n_coefficients_real(L_MAX)     # 4 for l_max = 1
    c_I = np.zeros(n_lm)
    c_I[0] = MONOPOLE                              # (0, 0)  monopole
    c_I[2] = DIPOLE_I                              # (1, 0)  intensity dipole (z)
    # V vector excludes the monopole: indices map to (1,-1), (1,0), (1,1)
    c_V = np.zeros(n_lm - 1)
    c_V[2] = DIPOLE_V                              # (1, 1)  V dipole (x, perp)
    return c_I, c_V


def main(args):
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(PLOTDIR, exist_ok=True)

    n_lm = sph.get_n_coefficients_real(L_MAX)
    n_lm_V = n_lm - 1
    c_I_inj, c_V_inj = injected_coefficients()

    # Parameter layout: [log_amplitude, tilt, c^I_{l>=1}, c^V_{l>=1}]
    names = ["log_amplitude", "tilt",
             "cI_1m1", "cI_10", "cI_1p1",
             "cV_1m1", "cV_10", "cV_1p1"]
    labels = [r"$\log_{10}A$", r"$\gamma$",
              r"$c^{I}_{1,-1}$", r"$c^{I}_{1,0}$", r"$c^{I}_{1,1}$",
              r"$c^{V}_{1,-1}$", r"$c^{V}_{1,0}$", r"$c^{V}_{1,1}$"]
    truths = np.concatenate(
        ([LOG_AMPLITUDE, TILT], c_I_inj[1:], c_V_inj)
    )
    ndim = len(names)

    print("=" * 66)
    print("  ell = 1 injection in intensity AND circular polarisation")
    print("=" * 66)
    print(f"  N_pulsars={args.n_pulsars}  T_obs={T_OBS_YRS}yr  "
          f"n_freq={args.n_frequencies}  l_max={L_MAX}  realization={REALIZATION}")
    print(f"  injected log_amplitude={LOG_AMPLITUDE}  tilt={TILT}")
    print(f"  injected c^I_1m = {np.array2string(c_I_inj[1:], precision=3)}")
    print(f"  injected c^V_1m = {np.array2string(c_V_inj, precision=3)}")

    # Precompute basis maps for the (fast) physical prior check
    Yp = real_ylm_basis(NSIDE_PRIOR, n_lm, L_MAX)

    # Injection sanity: positive intensity and |V| <= I everywhere (both hold
    # because C_l is defined relative to the monopole and the dipoles are on
    # perpendicular axes).
    I_inj = Yp @ c_I_inj
    V_inj = Yp @ np.concatenate([[0.0], c_V_inj])
    ratio = np.max(np.abs(V_inj) / I_inj)
    print(f"  C_l/C_0^I = {C_ELL:.3f}  ->  c_lm/c_00 = {COEFF:.3f}  "
          f"(c_lm = {DIPOLE_I:.4f})")
    print(f"  min I(n)={I_inj.min():.4e} (>0)   max|V|/I={ratio:.3f} (<1)")
    if I_inj.min() <= 0 or ratio >= 1.0:
        raise ValueError("Injection violates I>0 or |V|<=I; reduce C_ELL.")

    if args.dry_run:
        print("\n  DRY RUN — setup only.")
        return

    # ----- ingredients (same catalog for data and likelihood) -----
    frequency = (1.0 + jnp.arange(args.n_frequencies)) / (T_OBS_YRS * ut.yr)
    signal_std = np.sqrt(
        np.asarray(SIGNAL_MODEL.template(frequency, [LOG_AMPLITUDE, TILT]))
    )

    noiseless = ut.load_yaml(NOISELESS_YAML)
    generate_catalog_kwargs = {
        "n_pulsars": args.n_pulsars,
        "save_catalog": True,
        "use_ng_positions": False,
        **noiseless,
    }
    strain_omega, response_IJ_lm, _, _, response_IJ_V_lm = get_tensors(
        frequency,
        path_to_pulsar_catalog=CATALOG_PATH,
        regenerate_catalog=True,
        anisotropies=True,
        circ_pol=True,
        l_max=L_MAX,
        **generate_catalog_kwargs,
    )
    response_IJ_lm = np.asarray(response_IJ_lm)        # (n_lm,  F, N, N)
    response_IJ_V_lm = np.asarray(response_IJ_V_lm)    # (n_lm_V, F, N, N)
    strain_omega = np.asarray(strain_omega)

    # ----- inject the data with the complex-Hermitian covariance -----
    R_resp_inj = np.einsum("v,vfij->fij", c_I_inj, response_IJ_lm)
    A_resp_inj = np.einsum("v,vfij->fij", c_V_inj, response_IJ_V_lm)
    _, data, *_ = generate_MCMC_data(
        REALIZATION, frequency, signal_std, strain_omega,
        R_resp_inj, None, None,
        response_IJ_V=A_resp_inj, signal_std_V=signal_std,
        save_MCMC_data=False,
    )

    # ----- fully-JAX posterior (jit + grad compatible) sampling c_lm -----
    # Everything below is traceable: the lm-response contraction is jnp.einsum,
    # the I>0 / |V|<=I positivity prior uses a precomputed real-Y_lm basis and
    # jnp.where (no numpy / healpy in the hot loop), and the spectrum comes from
    # the (jit/grad-compatible) signal model template.
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

        # Box + physical priors as a single boolean mask (used as a where-gate)
        in_box = (
            (logA > LOGA_MIN) & (logA < LOGA_MAX)
            & (tilt > TILT_MIN) & (tilt < TILT_MAX)
            & (jnp.max(jnp.abs(cI)) < PRIOR) & (jnp.max(jnp.abs(cV)) < PRIOR)
        )
        I_map = Yp_j @ c_I_full
        V_map = Yp_j @ c_V_full
        ratio = jnp.max(jnp.abs(V_map) / jnp.where(I_map > 0, I_map, 1.0))
        phys = (jnp.min(I_map) > 0.0) & (ratio < 1.0)
        ok = in_box & phys

        R_resp = jnp.einsum("v,vfij->fij", c_I_full, resp_I_j)
        A_resp = jnp.einsum("v,vfij->fij", c_V_full[1:], resp_V_j)
        sv = SIGNAL_MODEL.template(frequency, jnp.array([logA, tilt]))
        ll = log_likelihood(
            data_j, sv, R_resp, strain_j,
            response_IJ_V=A_resp, signal_value_V=sv,
        )
        return jnp.where(ok, ll, -jnp.inf)

    # jit + grad of the full posterior (the refactor goal)
    logp_jit = jax.jit(log_posterior_jax)
    grad_jit = jax.jit(jax.grad(log_posterior_jax))
    lp0 = float(logp_jit(jnp.asarray(truths)))
    g0 = np.asarray(grad_jit(jnp.asarray(truths)))
    print(f"\n  [JAX posterior] jit logp(truth) = {lp0:.3f}")
    print(f"  [JAX posterior] grad finite = {np.all(np.isfinite(g0))}  "
          f"||grad|| = {np.linalg.norm(g0):.2e}")

    # Vectorised (jit + vmap over walkers) batched log-prob for emcee
    logp_batched = jax.jit(jax.vmap(log_posterior_jax))

    def logp_np(thetas):
        return np.asarray(logp_batched(jnp.asarray(thetas)))

    rng = np.random.default_rng(0)
    scale = np.concatenate(([0.1, 0.1], 0.02 * np.ones(2 * n_lm_V)))
    initial = truths[None, :] + scale[None, :] * rng.standard_normal(
        (args.n_walkers, ndim)
    )

    print(f"\n  Running vectorised emcee: {args.n_walkers} walkers, "
          f"{ndim} parameters, {args.burnin_steps} burn-in + up to "
          f"{args.i_max}x{args.mcmc_steps} steps "
          f"(until |R-1| < {R_CONVERGENCE:g}) ...")
    t0 = time.time()
    sampler = emcee.EnsembleSampler(
        args.n_walkers, ndim, logp_np, vectorize=True
    )
    state = sampler.run_mcmc(initial, args.burnin_steps, progress=True)
    sampler.reset()

    # Run in chunks of mcmc_steps, recomputing the Gelman-Rubin statistic after
    # each chunk, until R-1 drops below R_CONVERGENCE or i_max chunks elapse.
    # Matches the convention in fastPTA.MCMC_code (mean-squared R_criterion):
    # R = sqrt(mean(R_param**2)) over parameters, converged when |R-1| is small.
    R_minus_1 = np.inf
    R_array = np.full(ndim, np.nan)
    for i in range(args.i_max):
        state = sampler.run_mcmc(state, args.mcmc_steps, progress=True)
        R_array = ut.get_R(sampler.get_chain())            # R per parameter
        R = np.sqrt(np.mean(R_array ** 2))                 # mean-squared
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
    print(f"  sampling wall time: {time.time() - t0:.1f}s   "
          f"mean acceptance: {np.mean(sampler.acceptance_fraction):.2f}   "
          f"final R-1 = {R_minus_1:.4f}")
    np.savez(os.path.join(OUTDIR, f"injection_l1_IV_chains_{args.n_pulsars}Np_{args.n_frequencies}Nf.npz"),
             samples=samples, pdfs=pdfs, truths=truths, names=names,
             R_minus_1=R_minus_1, R_array=R_array)

    # ----- summary -----
    med = np.median(samples, axis=0)
    lo = np.quantile(samples, 0.16, axis=0)
    hi = np.quantile(samples, 0.84, axis=0)
    print("\n  Recovered (median [16,84]%) vs injected:")
    for i, name in enumerate(names):
        print(f"    {name:16s} = {med[i]:+.4f} [{lo[i]:+.4f}, {hi[i]:+.4f}]"
              f"   (injected {truths[i]:+.4f})")

    # ----- angular power in the paper convention: C_l / C_0^I -----
    # C_l = (1/(2l+1)) sum_m c_lm^2 (sph.get_CL_from_real_clm); the V power is
    # normalised by the intensity monopole C_0^I = |c^I_00|^2 = 1/(4 pi).
    C0_I = MONOPOLE ** 2
    c_I_med = np.concatenate([[MONOPOLE], med[2:2 + n_lm_V]])
    c_V_med = np.concatenate([[0.0], med[2 + n_lm_V:]])
    Cl_I_inj = sph.get_CL_from_real_clm(c_I_inj)[1:] / C0_I
    Cl_I_rec = sph.get_CL_from_real_clm(c_I_med)[1:] / C0_I
    Cl_V_inj = sph.get_CL_from_real_clm(
        np.concatenate([[0.0], c_V_inj]))[1:] / C0_I
    Cl_V_rec = sph.get_CL_from_real_clm(c_V_med)[1:] / C0_I
    print("\n  Angular power (paper convention, C_l / C_0^I):")
    for li, ell in enumerate(range(1, L_MAX + 1)):
        print(f"    C_{ell}^I/C_0^I = {Cl_I_rec[li]:.4f} (injected {Cl_I_inj[li]:.4f})"
              f"   C_{ell}^V/C_0^I = {Cl_V_rec[li]:.4f} (injected {Cl_V_inj[li]:.4f})")

    # ----- corner plot -----
    fig = corner.corner(samples, labels=labels, truths=truths,
                        truth_color="C3", show_titles=True,
                        title_fmt=".3f", title_kwargs={"fontsize": 8},
                        label_kwargs={"fontsize": 11})
    fig.suptitle("ell=1 injection: I + V parameter recovery", fontsize=13)
    corner_path = os.path.join(PLOTDIR, f"injection_l1_IV_corner_{args.n_pulsars}Np_{args.n_frequencies}Nf.png")
    fig.savefig(corner_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Corner plot saved to {corner_path}")

    # ----- injected vs reconstructed sky maps -----
    # (c_I_med, c_V_med computed above for the angular-power summary)
    I_inj_map = sph.get_map_from_real_clms(c_I_inj, NSIDE_PLOT, l_max=L_MAX)
    V_inj_map = sph.get_map_from_real_clms(
        np.concatenate([[0.0], c_V_inj]), NSIDE_PLOT, l_max=L_MAX)
    I_rec_map = sph.get_map_from_real_clms(c_I_med, NSIDE_PLOT, l_max=L_MAX)
    V_rec_map = sph.get_map_from_real_clms(c_V_med, NSIDE_PLOT, l_max=L_MAX)

    I_lim = (min(I_inj_map.min(), I_rec_map.min()),
             max(I_inj_map.max(), I_rec_map.max()))
    V_abs = max(np.abs(V_inj_map).max(), np.abs(V_rec_map).max())

    fig = plt.figure(figsize=(11, 7))
    hp.mollview(I_inj_map, sub=(2, 2, 1), title="Injected $I(\\hat n)$",
                cmap="viridis", min=I_lim[0], max=I_lim[1])
    hp.mollview(I_rec_map, sub=(2, 2, 2), title="Reconstructed $I(\\hat n)$",
                cmap="viridis", min=I_lim[0], max=I_lim[1])
    hp.mollview(V_inj_map, sub=(2, 2, 3), title="Injected $V(\\hat n)$",
                cmap="RdBu_r", min=-V_abs, max=V_abs)
    hp.mollview(V_rec_map, sub=(2, 2, 4), title="Reconstructed $V(\\hat n)$",
                cmap="RdBu_r", min=-V_abs, max=V_abs)
    map_path = os.path.join(PLOTDIR, f"injection_l1_IV_skymaps_{args.n_pulsars}Np_{args.n_frequencies}Nf.png")
    fig.savefig(map_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  Sky maps saved to {map_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pulsars", type=int, default=N_PULSARS)
    parser.add_argument("--n-frequencies", type=int, default=N_FREQUENCIES)
    parser.add_argument("--n-walkers", type=int, default=N_WALKERS)
    parser.add_argument("--i-max", type=int, default=I_MAX)
    parser.add_argument("--burnin-steps", type=int, default=BURNIN_STEPS)
    parser.add_argument("--mcmc-steps", type=int, default=MCMC_STEPS)
    parser.add_argument("--quick", action="store_true",
                        help="Small/fast smoke run.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the setup without running.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.n_pulsars = min(args.n_pulsars, 30)
        args.n_frequencies = 10
        args.n_walkers = 20
        args.i_max = 2
        args.burnin_steps = 50
        args.mcmc_steps = 50

    main(args)
