"""
run_injection_l1_IV_nuts.py
===========================
Same ell = 1 intensity + circular-polarisation injection/recovery as
``run_injection_l1_IV.py``, but sampled with gradient-based NUTS (numpyro)
instead of emcee. This exploits the fact that the data covariance / likelihood
(``C = R + i A`` with complex-Hermitian linear algebra) is fully jax.jit and
jax.grad compatible: NUTS uses the gradient of the log-posterior to explore far
more efficiently than the gradient-free ensemble sampler, so it needs many
fewer (expensive, N x N complex) likelihood evaluations for the same effective
sample size.

Requires numpyro (available in the ``ng20_gwb`` conda environment). Run with:

    /opt/homebrew/Caskroom/miniforge/base/envs/ng20_gwb/bin/python \
        run_injection_l1_IV_nuts.py [--n-pulsars 150] [--quick]
"""

import argparse
import os
import sys
import time

import numpy as np
import jax
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS, init_to_value

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

numpyro.enable_x64()
HERE = os.path.dirname(os.path.abspath(__file__))

# --------------------------- configuration ---------------------------
L_MAX = 1
N_PULSARS = 150
T_OBS_YRS = 16.03
N_FREQUENCIES = 12

SIGNAL_MODEL = get_signal_model("power_law")
LOG_AMPLITUDE = -7.1995
TILT = 2.0

# Injected ell=1 mode in the PAPER CONVENTION (see Cl_conventions_and_priors.md):
#   C_l = (1/(2l+1)) sum_m c_lm^2,   C_0 = |c_00|^2 = 1/(4 pi)   (get_CL_from_real_clm)
# A single m is excited, so c_lm = sqrt((2l+1)*C_ELL)*c_00; the V power is
# normalised by the intensity monopole C_0^I (there is no C_0^V), i.e. C_ELL is
# C_1^I/C_0^I for I and C_1^V/C_0^I for V.
C_ELL = 0.05                             # C_l / C_0^I for the injected ell=1 mode
MONOPOLE = 1.0 / np.sqrt(4.0 * np.pi)
COEFF = np.sqrt((2 * L_MAX + 1) * C_ELL)  # c_lm / c_00 for a single excited m
DIPOLE_I = COEFF * MONOPOLE               # intensity c^I_{1,0} (z-axis)
DIPOLE_V = COEFF * MONOPOLE               # V-mode    c^V_{1,1} (x-axis, perp)
PRIOR = 5.0 / (4.0 * np.pi)

LOGA_MIN, LOGA_MAX = -8.5, -5.5
TILT_MIN, TILT_MAX = 1.0, 3.0
REALIZATION = False

NUM_WARMUP = 500
NUM_SAMPLES = 1000
NSIDE_PRIOR = 8
NSIDE_PLOT = 64

OUTDIR = os.path.join(HERE, "generated_data")
PLOTDIR = os.path.join(HERE, "plots")
CATALOG_PATH = os.path.join(HERE, "..", "pulsar_configurations",
                            "tmp_injection_l1_IV_nuts.txt")
NOISELESS_YAML = os.path.join(HERE, "..", "pulsar_configurations",
                              "EPTAlike_pulsar_parameters_noiseless.yaml")


def real_ylm_basis(nside, n_lm, l_max):
    npix = hp.nside2npix(nside)
    Y = np.zeros((npix, n_lm))
    for k in range(n_lm):
        e = np.zeros(n_lm)
        e[k] = 1.0
        Y[:, k] = sph.get_map_from_real_clms(e, nside, l_max=l_max)
    return Y


def main(args):
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(PLOTDIR, exist_ok=True)

    n_lm = sph.get_n_coefficients_real(L_MAX)
    n_lm_V = n_lm - 1
    c_I_inj = np.zeros(n_lm); c_I_inj[0] = MONOPOLE; c_I_inj[2] = DIPOLE_I
    c_V_inj = np.zeros(n_lm_V); c_V_inj[2] = DIPOLE_V

    names = ["log_amplitude", "tilt",
             "cI_1m1", "cI_10", "cI_1p1", "cV_1m1", "cV_10", "cV_1p1"]
    labels = [r"$\log_{10}A$", r"$\gamma$",
              r"$c^{I}_{1,-1}$", r"$c^{I}_{1,0}$", r"$c^{I}_{1,1}$",
              r"$c^{V}_{1,-1}$", r"$c^{V}_{1,0}$", r"$c^{V}_{1,1}$"]
    truths = np.concatenate(([LOG_AMPLITUDE, TILT], c_I_inj[1:], c_V_inj))

    print("=" * 66)
    print("  ell=1 I+V injection  ->  NUTS (numpyro) recovery")
    print("=" * 66)
    print(f"  N_pulsars={args.n_pulsars}  n_freq={args.n_frequencies}  "
          f"l_max={L_MAX}  C_l/C_0^I={C_ELL}")

    Yp = real_ylm_basis(NSIDE_PRIOR, n_lm, L_MAX)
    Yp_j = jnp.asarray(Yp)

    # ----- ingredients (same catalog for data and likelihood) -----
    frequency = (1.0 + jnp.arange(args.n_frequencies)) / (T_OBS_YRS * ut.yr)
    signal_std = np.sqrt(
        np.asarray(SIGNAL_MODEL.template(frequency, [LOG_AMPLITUDE, TILT])))
    noiseless = ut.load_yaml(NOISELESS_YAML)
    strain_omega, response_IJ_lm, _, _, response_IJ_V_lm = get_tensors(
        frequency, path_to_pulsar_catalog=CATALOG_PATH, regenerate_catalog=True,
        anisotropies=True, circ_pol=True, l_max=L_MAX,
        n_pulsars=args.n_pulsars, save_catalog=True, use_ng_positions=False,
        **noiseless)
    resp_I_j = jnp.asarray(np.asarray(response_IJ_lm))
    resp_V_j = jnp.asarray(np.asarray(response_IJ_V_lm))
    strain_j = jnp.asarray(np.asarray(strain_omega))

    # ----- inject the data (complex Hermitian covariance) -----
    R_resp_inj = np.einsum("v,vfij->fij", c_I_inj, np.asarray(response_IJ_lm))
    A_resp_inj = np.einsum("v,vfij->fij", c_V_inj, np.asarray(response_IJ_V_lm))
    _, data, *_ = generate_MCMC_data(
        REALIZATION, frequency, signal_std, strain_omega, R_resp_inj, None,
        None, response_IJ_V=A_resp_inj, signal_std_V=signal_std,
        save_MCMC_data=False)
    data_j = jnp.asarray(data)

    # ----- numpyro model: priors + complex-Hermitian Whittle likelihood -----
    def model():
        logA = numpyro.sample("log_amplitude", dist.Uniform(LOGA_MIN, LOGA_MAX))
        tilt = numpyro.sample("tilt", dist.Uniform(TILT_MIN, TILT_MAX))
        cI = numpyro.sample(
            "cI", dist.Uniform(-PRIOR, PRIOR).expand([n_lm_V]).to_event(1))
        cV = numpyro.sample(
            "cV", dist.Uniform(-PRIOR, PRIOR).expand([n_lm_V]).to_event(1))

        c_I_full = jnp.concatenate([jnp.array([MONOPOLE]), cI])

        # Soft physical barrier: penalise I(n)<0 and |V(n)|>I(n) (kept smooth so
        # NUTS gradients stay well-defined; inactive near the physical injection)
        I_map = Yp_j @ c_I_full
        V_map = Yp_j @ jnp.concatenate([jnp.array([0.0]), cV])
        penalty = (jnp.sum(jax.nn.relu(-I_map))
                   + jnp.sum(jax.nn.relu(jnp.abs(V_map) - I_map)))
        numpyro.factor("phys", -1.0e4 * penalty)

        R_resp = jnp.einsum("v,vfij->fij", c_I_full, resp_I_j)
        A_resp = jnp.einsum("v,vfij->fij", cV, resp_V_j)
        sv = SIGNAL_MODEL.template(frequency, jnp.array([logA, tilt]))
        ll = log_likelihood(data_j, sv, R_resp, strain_j,
                            response_IJ_V=A_resp, signal_value_V=sv)
        numpyro.factor("loglike", ll)

    init = init_to_value(values={
        "log_amplitude": LOG_AMPLITUDE, "tilt": TILT,
        "cI": jnp.asarray(c_I_inj[1:]), "cV": jnp.asarray(c_V_inj)})
    kernel = NUTS(model, target_accept_prob=0.9, init_strategy=init)
    mcmc = MCMC(kernel, num_warmup=args.num_warmup, num_samples=args.num_samples,
                num_chains=1, progress_bar=not args.quiet)

    print(f"\n  Running NUTS: {args.num_warmup} warmup + {args.num_samples} "
          f"samples ...")
    t0 = time.time()
    mcmc.run(jax.random.PRNGKey(0))
    wall = time.time() - t0
    print(f"  NUTS wall time: {wall:.1f}s")
    if not args.quiet:
        mcmc.print_summary()

    s = mcmc.get_samples()
    samples = np.column_stack([
        np.asarray(s["log_amplitude"]), np.asarray(s["tilt"]),
        np.asarray(s["cI"]), np.asarray(s["cV"])])
    np.savez(os.path.join(OUTDIR, "injection_l1_IV_nuts_chains.npz"),
             samples=samples, truths=truths, names=names, wall_time=wall)

    med = np.median(samples, axis=0)
    lo = np.quantile(samples, 0.16, axis=0)
    hi = np.quantile(samples, 0.84, axis=0)
    print("\n  Recovered (median [16,84]%) vs injected:")
    for i, name in enumerate(names):
        print(f"    {name:16s} = {med[i]:+.4f} [{lo[i]:+.4f}, {hi[i]:+.4f}]"
              f"   (injected {truths[i]:+.4f})")

    # ----- corner -----
    fig = corner.corner(samples, labels=labels, truths=truths,
                        truth_color="C3", show_titles=True, title_fmt=".3f",
                        title_kwargs={"fontsize": 8},
                        label_kwargs={"fontsize": 11})
    fig.suptitle(f"ell=1 I+V recovery (NUTS, {args.n_pulsars} pulsars)",
                 fontsize=13)
    cpath = os.path.join(PLOTDIR, "injection_l1_IV_nuts_corner.png")
    fig.savefig(cpath, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"\n  Corner plot saved to {cpath}")

    # ----- injected vs reconstructed sky maps -----
    c_I_med = np.concatenate([[MONOPOLE], med[2:2 + n_lm_V]])
    c_V_med = np.concatenate([[0.0], med[2 + n_lm_V:]])
    I_inj = sph.get_map_from_real_clms(c_I_inj, NSIDE_PLOT, l_max=L_MAX)
    V_inj = sph.get_map_from_real_clms(np.concatenate([[0.0], c_V_inj]),
                                       NSIDE_PLOT, l_max=L_MAX)
    I_rec = sph.get_map_from_real_clms(c_I_med, NSIDE_PLOT, l_max=L_MAX)
    V_rec = sph.get_map_from_real_clms(c_V_med, NSIDE_PLOT, l_max=L_MAX)
    Ilim = (min(I_inj.min(), I_rec.min()), max(I_inj.max(), I_rec.max()))
    Vabs = max(np.abs(I_inj * 0 + V_inj).max(), np.abs(V_rec).max())

    fig = plt.figure(figsize=(11, 7))
    hp.mollview(I_inj, sub=(2, 2, 1), title="Injected $I(\\hat n)$",
                cmap="viridis", min=Ilim[0], max=Ilim[1])
    hp.mollview(I_rec, sub=(2, 2, 2), title="Reconstructed $I(\\hat n)$",
                cmap="viridis", min=Ilim[0], max=Ilim[1])
    hp.mollview(V_inj, sub=(2, 2, 3), title="Injected $V(\\hat n)$",
                cmap="RdBu_r", min=-Vabs, max=Vabs)
    hp.mollview(V_rec, sub=(2, 2, 4), title="Reconstructed $V(\\hat n)$",
                cmap="RdBu_r", min=-Vabs, max=Vabs)
    mpath = os.path.join(PLOTDIR, "injection_l1_IV_nuts_skymaps.png")
    fig.savefig(mpath, dpi=140, bbox_inches="tight"); plt.close(fig)
    print(f"  Sky maps saved to {mpath}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-pulsars", type=int, default=N_PULSARS)
    p.add_argument("--n-frequencies", type=int, default=N_FREQUENCIES)
    p.add_argument("--num-warmup", type=int, default=NUM_WARMUP)
    p.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    p.add_argument("--quick", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_pulsars = 30
        args.n_frequencies = 8
        args.num_warmup = 150
        args.num_samples = 300
    main(args)
