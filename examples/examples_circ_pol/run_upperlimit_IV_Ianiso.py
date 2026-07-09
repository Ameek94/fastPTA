"""
run_upperlimit_IV_Ianiso.py
===========================
Noiseless (CVL) Fisher forecast with a NON-ZERO INTENSITY ANISOTROPY injected and
ZERO CIRCULAR POLARISATION. A single intensity multipole (default the ell = 1
dipole) is injected at a target C_l^I/C_0^I, while the V sky is left isotropic
(zero V everywhere). The script then:

  * MEASURES C_l^I/C_0^I around the injection (95% interval), and
  * sets 95% UPPER LIMITS on C_l^V/C_0^I.

This probes how an intensity anisotropy affects the circular-polarisation
sensitivity. The I and V Fisher blocks decouple at zero-V injection (the I-V cross
Fisher vanishes), but the V auto-block still depends on the injected I anisotropy
through the covariance C = R(c^I), so the V limits can shift relative to the
isotropic case (run_upperlimit_IV_isotropic.py).

Conventions follow Cl_conventions_and_priors.md:
  * c^I_00 = 1/sqrt(4 pi)  ->  C_0^I = 1/(4 pi);  C_l = (1/(2l+1)) sum_m c_lm^2
  * single excited m: c_lm = sqrt((2l+1) * C_l/C_0^I) * c_00
  * V monopole excluded; V power normalised by the intensity monopole C_0^I
  * limits reported as C_l / C_0^I, flat box prior |c_lm| < 5/(4 pi)

Usage
-----
    python run_upperlimit_IV_Ianiso.py
    python run_upperlimit_IV_Ianiso.py --inj-l 1 --inj-cl 0.05 --n-pulsars 100
    python run_upperlimit_IV_Ianiso.py --dry-run
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

# Injected intensity anisotropy: single (INJ_L, m=0) mode at C_l^I/C_0^I = INJ_CL.
# Default dipole at 0.05 (below the maximal physical dipole C_1/C_0 = 1/9).
INJ_L = 1
INJ_CL = 0.05

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

    # ---- injection: I anisotropy (single m=0 mode), zero V ----
    signal_lm = cm.isotropic_signal_lm(args.l_max)
    signal_lm = cm.add_single_m_mode(
        signal_lm, l=args.inj_l, m=0, Cl_over_C0=args.inj_cl, l_max=args.l_max
    )
    signal_lm_V = np.zeros(n_lm_V)
    means_I = np.concatenate((cm.SIGNAL_PARAMETERS, signal_lm[1:]))
    means_V = signal_lm_V

    ell = np.arange(1, args.l_max + 1)

    # injected C_l^I/C_0^I per l (nonzero only at inj_l)
    injected_I = sph.get_CL_from_real_clm(signal_lm)[1:] / cm.C0_I

    print("=" * 66)
    print("  I+V upper limits — I ANISOTROPY injected, ZERO V")
    print("=" * 66)
    print(f"  N_pulsars={args.n_pulsars}  N_realizations={args.n_realizations}  "
          f"l_max={args.l_max}  T_obs={cm.T_OBS_YRS}yr  n_freq={cm.N_FREQUENCIES}")
    print(f"  injected intensity: ell={args.inj_l} (m=0) at "
          f"C_{args.inj_l}^I/C_0^I = {args.inj_cl}")
    print(f"  injected C_l^I/C_0^I = "
          f"{np.array2string(injected_I, precision=4)}")
    print(f"  injected C_l^V/C_0^I = 0  (isotropic V)")

    if args.dry_run:
        print("\n  DRY RUN — setup only.")
        return

    # ---- Fisher blocks ----
    fisher_I, fisher_V = cm.fisher_blocks(
        args.n_pulsars, signal_lm, signal_lm_V, args.l_max,
        args.n_realizations, catalog_tag="UL_Ianiso",
    )
    cov_I = ut.compute_inverse(fisher_I)
    cov_V = ut.compute_inverse(fisher_V)

    # ---- C_l / C_0^I: I measured around injection, V upper-limited ----
    Cl_I, Cl_I_prior = cm.cl_limits_I(means_I, cov_I)
    Cl_V, Cl_V_prior = cm.cl_limits_V(means_V, cov_V)
    Cl_I_ana = np.array([cm.cl_analytical(cov_I[j], args.l_max, cm.SHAPE_PARAMS)
                         for j in range(args.n_realizations)])
    Cl_V_ana = np.array([cm.cl_analytical(cov_V[j], args.l_max, 0)
                         for j in range(args.n_realizations)])

    # ---- summary ----
    print("\n  Results (mean over realizations, C_l / C_0^I):")
    print(f"    {'l':>3}  {'C_l^I (95%)':>14}  {'injected I':>12}  "
          f"{'C_l^V UL (95%)':>16}")
    for li, l in enumerate(ell):
        print(f"    {l:>3}  {np.mean(Cl_I[:, li]):>14.4e}  "
              f"{injected_I[li]:>12.4e}  {np.mean(Cl_V[:, li]):>16.4e}")

    # ---- save ----
    outfile = os.path.join(
        OUTDIR, f"UL_Ianiso_l{args.inj_l}_Npsr{args.n_pulsars}.npz")
    np.savez(outfile, ell=ell, Cl_I=Cl_I, Cl_I_prior=Cl_I_prior,
             Cl_V=Cl_V, Cl_V_prior=Cl_V_prior, Cl_I_ana=Cl_I_ana,
             Cl_V_ana=Cl_V_ana, injected_I=injected_I,
             inj_l=args.inj_l, inj_cl=args.inj_cl, n_pulsars=args.n_pulsars)
    print(f"\n  Saved {outfile}")

    # ---- plot ----
    cm.plot_limits(
        ell, Cl_I, Cl_V,
        title=f"I+V: ell={args.inj_l} intensity injection, zero V "
              f"({args.n_pulsars} pulsars, noiseless)",
        outpath=os.path.join(
            PLOTDIR, f"UL_Ianiso_l{args.inj_l}_Npsr{args.n_pulsars}.png"),
        injected_I=injected_I, ng15_dat=NG15_DAT,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pulsars", type=int, default=N_PULSARS)
    parser.add_argument("--n-realizations", type=int, default=N_REALIZATIONS)
    parser.add_argument("--l-max", type=int, default=L_MAX)
    parser.add_argument("--inj-l", type=int, default=INJ_L,
                        help="Multipole of the injected intensity anisotropy.")
    parser.add_argument("--inj-cl", type=float, default=INJ_CL,
                        help="Injected C_l^I / C_0^I for the (inj-l, m=0) mode.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the setup without running.")
    args = parser.parse_args()
    main(args)
