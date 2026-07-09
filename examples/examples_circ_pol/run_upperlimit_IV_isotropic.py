"""
run_upperlimit_IV_isotropic.py
==============================
Noiseless (CVL) Fisher 95% UPPER LIMITS on the I and V angular power spectra for a
ZERO-ANISOTROPY injection: the injected background is isotropic in BOTH intensity
(only the monopole c^I_00 = 1/sqrt(4 pi)) and circular polarisation (no V at all).
This is the I+V analogue of the intensity-only forecast in
examples_paper_anisotropies (arXiv:2407.14460) and the per-N sweep in
run_CVL_vs_npulsars.py, here at a single fixed array configuration.

Conventions follow Cl_conventions_and_priors.md:
  * c^I_00 = 1/sqrt(4 pi)  ->  C_0^I = 1/(4 pi)
  * C_l = (1/(2l+1)) sum_m c_lm^2  (sph.get_CL_from_real_clm)
  * V monopole excluded; V power normalised by the intensity monopole C_0^I
  * limits reported as C_l^I / C_0^I and C_l^V / C_0^I, flat box prior |c_lm| < 5/(4 pi)

Usage
-----
    python run_upperlimit_IV_isotropic.py
    python run_upperlimit_IV_isotropic.py --n-pulsars 100 --l-max 4
    python run_upperlimit_IV_isotropic.py --dry-run
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)

import fastPTA.utils as ut
from fastPTA.angular_decomposition import spherical_harmonics as sph

import upperlimit_IV_common as cm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

N_PULSARS = 100
N_REALIZATIONS = 30
L_MAX = 6

OUTDIR = os.path.join(HERE, "generated_data")
PLOTDIR = os.path.join(HERE, "plots")
NG15_DAT = os.path.join(
    HERE, "..", "examples_paper_anisotropies", "data_paper_2",
    "limits_Cl_powerlaw_lin_ng15.dat")


def main(args):
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(PLOTDIR, exist_ok=True)

    n_lm = sph.get_n_coefficients_real(args.l_max)
    n_lm_V = n_lm - 1

    # ---- zero-anisotropy injection: isotropic I, zero V ----
    signal_lm = cm.isotropic_signal_lm(args.l_max)   # monopole only
    signal_lm_V = np.zeros(n_lm_V)
    means_I = np.concatenate((cm.SIGNAL_PARAMETERS, signal_lm[1:]))  # zeros for l>=1
    means_V = signal_lm_V

    ell = np.arange(1, args.l_max + 1)

    print("=" * 66)
    print("  I+V upper limits — ZERO I and V anisotropy (isotropic injection)")
    print("=" * 66)
    print(f"  N_pulsars={args.n_pulsars}  N_realizations={args.n_realizations}  "
          f"l_max={args.l_max}  T_obs={cm.T_OBS_YRS}yr  n_freq={cm.N_FREQUENCIES}")
    print(f"  injected: C_l^I/C_0^I = 0 (l>=1),  C_l^V/C_0^I = 0  (isotropic)")

    if args.dry_run:
        print("\n  DRY RUN — setup only.")
        return

    # ---- Fisher blocks ----
    fisher_I, fisher_V = cm.fisher_blocks(
        args.n_pulsars, signal_lm, signal_lm_V, args.l_max,
        args.n_realizations, catalog_tag="UL_iso",
    )
    cov_I = ut.compute_inverse(fisher_I)
    cov_V = ut.compute_inverse(fisher_V)

    # ---- C_l / C_0^I upper limits ----
    Cl_I, Cl_I_prior = cm.cl_limits_I(means_I, cov_I)
    Cl_V, Cl_V_prior = cm.cl_limits_V(means_V, cov_V)
    Cl_I_ana = np.array([cm.cl_analytical(cov_I[j], args.l_max, cm.SHAPE_PARAMS)
                         for j in range(args.n_realizations)])
    Cl_V_ana = np.array([cm.cl_analytical(cov_V[j], args.l_max, 0)
                         for j in range(args.n_realizations)])

    # ---- summary ----
    print("\n  95% upper limits (mean over realizations, C_l / C_0^I):")
    print(f"    {'l':>3}  {'C_l^I/C_0^I':>14}  {'C_l^V/C_0^I':>14}")
    for li, l in enumerate(ell):
        print(f"    {l:>3}  {np.mean(Cl_I[:, li]):>14.4e}  "
              f"{np.mean(Cl_V[:, li]):>14.4e}")

    # ---- save ----
    outfile = os.path.join(OUTDIR, f"UL_iso_Npsr{args.n_pulsars}.npz")
    np.savez(outfile, ell=ell, Cl_I=Cl_I, Cl_I_prior=Cl_I_prior,
             Cl_V=Cl_V, Cl_V_prior=Cl_V_prior, Cl_I_ana=Cl_I_ana,
             Cl_V_ana=Cl_V_ana, n_pulsars=args.n_pulsars)
    print(f"\n  Saved {outfile}")

    # ---- plot ----
    cm.plot_limits(
        ell, Cl_I, Cl_V,
        title=f"I+V 95% upper limits — isotropic injection "
              f"({args.n_pulsars} pulsars, noiseless)",
        outpath=os.path.join(PLOTDIR, f"UL_iso_Npsr{args.n_pulsars}.png"),
        ng15_dat=NG15_DAT,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pulsars", type=int, default=N_PULSARS)
    parser.add_argument("--n-realizations", type=int, default=N_REALIZATIONS)
    parser.add_argument("--l-max", type=int, default=L_MAX)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the setup without running.")
    args = parser.parse_args()
    main(args)
