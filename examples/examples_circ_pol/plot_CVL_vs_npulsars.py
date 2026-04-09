"""
plot_CVL_vs_npulsars.py
=======================
Plot the results produced by run_CVL_vs_npulsars.py.

Produces three figures saved in plots/:

  Fig 1 — CVL_scaling_I.pdf
      C_l^I 95% detectability threshold vs ell for each N_pulsars,
      alongside the NANOGrav 15-yr noisy upper limits.

  Fig 2 — CVL_scaling_V.pdf
      C_l^V 95% detectability threshold vs ell for each N_pulsars,
      compared to the analytical chi^2 thresholds (cross-check).

  Fig 3 — CVL_scaling_comparison.pdf
      C_l^V / C_l^I ratio vs N_pulsars for each multipole l,
      showing how the V/I sensitivity gap evolves with array size.

Usage
-----
    python plot_CVL_vs_npulsars.py
    python plot_CVL_vs_npulsars.py --indir generated_data --outdir plots
"""

import argparse
import glob
import os
import re

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INDIR  = "generated_data"
OUTDIR = "plots"

# Reference NANOGrav 15-yr intensity limits
NG15_DAT = "../examples_paper_anisotropies/data_paper_2/limits_Cl_powerlaw_lin_ng15.dat"

# Colour cycle for different N_pulsars (using matplotlib tableau palette)
COLORS = [
    "tab:blue", "tab:orange", "tab:green", "tab:red",
    "tab:purple", "tab:brown", "tab:pink", "tab:gray",
]

# Plot style
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_per_n(indir):
    """
    Scan indir for all CVL_Npsr{N}.npz files and return an ordered dict
    mapping N_pulsars (int) -> loaded npz object.
    """
    pattern = os.path.join(indir, "CVL_Npsr*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(
            f"No CVL_Npsr*.npz files found in '{indir}'.\n"
            "Run run_CVL_vs_npulsars.py first."
        )
    per_n = {}
    for f in files:
        m = re.search(r"CVL_Npsr(\d+)\.npz$", f)
        if m:
            n = int(m.group(1))
            per_n[n] = np.load(f)
    return dict(sorted(per_n.items()))


def build_summary_from_per_n(per_n):
    """Aggregate per-N data into the same structure as CVL_summary.npz."""
    n_arr = np.array(sorted(per_n.keys()))
    ell   = per_n[n_arr[0]]["ell"]
    mean_Cl_I     = np.array([np.mean(per_n[n]["Cl_I"],     axis=0) for n in n_arr])
    mean_Cl_V     = np.array([np.mean(per_n[n]["Cl_V"],     axis=0) for n in n_arr])
    mean_Cl_I_ana = np.array([np.mean(per_n[n]["Cl_I_ana"], axis=0) for n in n_arr])
    mean_Cl_V_ana = np.array([np.mean(per_n[n]["Cl_V_ana"], axis=0) for n in n_arr])
    return {
        "n_pulsars":     n_arr,
        "ell":           ell,
        "mean_Cl_I":     mean_Cl_I,
        "mean_Cl_V":     mean_Cl_V,
        "mean_Cl_I_ana": mean_Cl_I_ana,
        "mean_Cl_V_ana": mean_Cl_V_ana,
    }


def axis_style(ax):
    """Apply common tick / spine style."""
    ax.tick_params(axis="both", which="both", direction="in", width=0.6)
    ax.xaxis.set_ticks_position("both")
    ax.yaxis.set_ticks_position("both")
    for spine in ax.spines.values():
        spine.set_linewidth(0.6)


def plot_errorbar(ax, x, data_2d, color, label, marker=".", offset=0.0, **kw):
    """
    Plot mean ± 95% quantile interval from a (N_realizations, n_ell) array.
    offset shifts x slightly for visual clarity.
    """
    mean = np.mean(data_2d, axis=0)
    lo   = mean - np.quantile(data_2d, 0.025, axis=0)
    hi   = np.quantile(data_2d, 0.975, axis=0) - mean
    ax.errorbar(x + offset, mean, yerr=(lo, hi),
                color=color, label=label, marker=marker,
                capsize=3, lw=1.2, **kw)
    return mean


# ---------------------------------------------------------------------------
# Figure 1 — C_l^I vs ell for each N_pulsars
# ---------------------------------------------------------------------------

def fig_Cl_I(summary, per_n, ell, ng15, outdir):
    n_arr = summary["n_pulsars"]
    n_ell = len(ell)

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=150)
    axis_style(ax)

    # NANOGrav 15-yr reference
    ax.semilogy(ell, ng15, color="black", marker="x", lw=0,
                ms=7, label="NANOGrav 15-yr (noisy)", zorder=5)

    for i, (n, color) in enumerate(zip(n_arr, COLORS)):
        data = per_n.get(n)
        if data is None:
            ax.semilogy(ell, summary["mean_Cl_I"][i], color=color,
                        marker=".", label=f"$N_p={n}$ (CVL)")
        else:
            plot_errorbar(ax, ell, data["Cl_I"], color=color,
                          label=f"$N_p={n}$ (CVL)")

    box = dict(boxstyle="round", facecolor="white", alpha=0.9,
               linewidth=0.8, edgecolor="0.7")
    ax.text(0.97, 0.97, "Noiseless (CVL)",
            transform=ax.transAxes, fontsize=8, va="top", ha="right", bbox=box)

    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$C_\ell^I \;/\; C_0^I$  (95% upper limit)")
    ax.set_title(r"Intensity anisotropy detectability — CVL")
    ax.set_xticks(ell)
    ax.legend(loc="lower right")
    ax.set_ylim(bottom=1e-4)

    plt.tight_layout()
    out = os.path.join(outdir, "CVL_scaling_I.pdf")
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2 — C_l^V vs ell for each N_pulsars
# ---------------------------------------------------------------------------

def fig_Cl_V(summary, per_n, ell, outdir):
    n_arr = summary["n_pulsars"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150,
                             sharey=False)
    axis_style(axes[0])
    axis_style(axes[1])

    # Left: C_l^V MC limits per N_pulsars
    ax = axes[0]
    for i, (n, color) in enumerate(zip(n_arr, COLORS)):
        data = per_n.get(n)
        if data is None:
            ax.semilogy(ell, summary["mean_Cl_V"][i], color=color,
                        marker="s", label=f"$N_p={n}$")
        else:
            plot_errorbar(ax, ell, data["Cl_V"], color=color,
                          label=f"$N_p={n}$", marker="s")

    box = dict(boxstyle="round", facecolor="white", alpha=0.9,
               linewidth=0.8, edgecolor="0.7")
    ax.text(0.97, 0.97, "Noiseless (CVL)",
            transform=ax.transAxes, fontsize=8, va="top", ha="right", bbox=box)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$C_\ell^V \;/\; C_0$  (95% upper limit)")
    ax.set_title(r"Circular polarization detectability — CVL")
    ax.set_xticks(ell)
    ax.legend(loc="lower right")
    ax.set_ylim(bottom=1e-4)

    # Right: MC / analytical ratio (should be ~1 if posterior is Gaussian)
    ax = axes[1]
    axis_style(ax)
    ax.axhline(1.0, color="black", lw=0.8, ls="--", label="Ideal (Gaussian)")
    for i, (n, color) in enumerate(zip(n_arr, COLORS)):
        data = per_n.get(n)
        if data is None:
            continue
        ratio = np.mean(data["Cl_V"], axis=0) / np.mean(data["Cl_V_ana"], axis=0)
        ax.plot(ell, ratio, color=color, marker="s", lw=1.2,
                label=f"$N_p={n}$")

    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$\langle C_\ell^V\rangle_{\rm MC} \;/\; C_\ell^{V,\,\rm ana}$")
    ax.set_title(r"MC / Analytical ratio ($C_\ell^V$)")
    ax.set_xticks(ell)
    ax.legend(loc="best")

    plt.tight_layout()
    out = os.path.join(outdir, "CVL_scaling_V.pdf")
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3 — C_l^V / C_l^I ratio vs N_pulsars for each ell
# ---------------------------------------------------------------------------

def fig_ratio_vs_N(summary, per_n, ell, outdir):
    n_arr   = summary["n_pulsars"]
    l_max   = len(ell)

    # Build colour map over multipoles
    cmap   = plt.get_cmap("plasma", l_max)
    colors = [cmap(i) for i in range(l_max)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=150)

    # Left: V/I ratio vs N_pulsars per multipole
    ax = axes[0]
    axis_style(ax)
    for li, (l_val, color) in enumerate(zip(ell, colors)):
        ratios = []
        for n in n_arr:
            data = per_n.get(n)
            if data is None:
                ratios.append(summary["mean_Cl_V"][list(n_arr).index(n), li]
                              / summary["mean_Cl_I"][list(n_arr).index(n), li])
            else:
                ratios.append(
                    np.mean(data["Cl_V"], axis=0)[li]
                    / np.mean(data["Cl_I"], axis=0)[li]
                )
        ax.plot(n_arr, ratios, color=color, marker="o", lw=1.5,
                label=rf"$\ell={int(l_val)}$")

    ax.set_xlabel(r"$N_{\rm pulsars}$")
    ax.set_ylabel(r"$\langle C_\ell^V \rangle \;/\; \langle C_\ell^I \rangle$")
    ax.set_title(r"$C_\ell^V / C_\ell^I$ sensitivity ratio — CVL")
    ax.legend(loc="best", ncol=2)
    ax.set_xscale("log")

    # Right: absolute C_l^I and C_l^V vs N_pulsars for each multipole
    ax = axes[1]
    axis_style(ax)
    for li, (l_val, color) in enumerate(zip(ell, colors)):
        mean_I = np.array([
            np.mean(per_n[n]["Cl_I"], axis=0)[li] if per_n.get(n) is not None
            else summary["mean_Cl_I"][list(n_arr).index(n), li]
            for n in n_arr
        ])
        mean_V = np.array([
            np.mean(per_n[n]["Cl_V"], axis=0)[li] if per_n.get(n) is not None
            else summary["mean_Cl_V"][list(n_arr).index(n), li]
            for n in n_arr
        ])
        kw = dict(lw=1.5, color=color)
        ax.loglog(n_arr, mean_I, marker=".", ls="-",
                  label=rf"$C_{{\ell={int(l_val)}}}^I$" if li == 0 else f"I, l={int(l_val)}",
                  **kw)
        ax.loglog(n_arr, mean_V, marker="s", ls="--",
                  label=rf"$C_{{\ell={int(l_val)}}}^V$" if li == 0 else f"V, l={int(l_val)}",
                  **kw)

    # Indicate N^{-1} scaling
    n_ref = np.array([n_arr[0], n_arr[-1]], dtype=float)
    scale = np.mean(
        [np.mean(per_n[n_arr[0]]["Cl_I"], axis=0) if per_n.get(n_arr[0]) is not None
         else summary["mean_Cl_I"][0]]
    )
    ax.loglog(n_ref, scale * (n_ref[0] / n_ref), color="grey",
              lw=0.8, ls=":", label=r"$\propto N^{-1}$")

    ax.set_xlabel(r"$N_{\rm pulsars}$")
    ax.set_ylabel(r"$C_\ell \;/\; C_0$  (95% CVL threshold)")
    ax.set_title(r"CVL thresholds vs. array size")
    # Compact legend: only first I, first V, and reference slope
    handles, labels = ax.get_legend_handles_labels()
    # Show one I, one V, and the N^{-1} reference
    shown = [h for h, l in zip(handles, labels) if "l=1" in l or "N^{-1}" in l or "I, l=1" in l or "V, l=1" in l]
    ax.legend(handles[:2] + handles[-1:], labels[:2] + labels[-1:],
              loc="best", fontsize=8)

    plt.tight_layout()
    out = os.path.join(outdir, "CVL_scaling_comparison.pdf")
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4 — combined I and V on one panel, all N_pulsars
# ---------------------------------------------------------------------------

def fig_combined(summary, per_n, ell, ng15, outdir):
    n_arr = summary["n_pulsars"]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    axis_style(ax)

    ax.semilogy(ell, ng15, color="black", marker="x", lw=0, ms=7,
                label="NANOGrav 15-yr $C_\ell^I$ (noisy)", zorder=6)

    for i, (n, color) in enumerate(zip(n_arr, COLORS)):
        data = per_n.get(n)
        if data is None:
            ax.semilogy(ell, summary["mean_Cl_I"][i], color=color,
                        marker=".", ls="-", label=f"$N_p={n}$, I")
            ax.semilogy(ell, summary["mean_Cl_V"][i], color=color,
                        marker="s", ls="--", label=f"$N_p={n}$, V")
        else:
            plot_errorbar(ax, ell, data["Cl_I"], color=color,
                          label=f"$N_p={n}$, I")
            plot_errorbar(ax, ell, data["Cl_V"], color=color,
                          label=f"$N_p={n}$, V", marker="s", ls="--")

    ax.legend(loc="lower right", fontsize=8, ncol=2, handlelength=1.5)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$C_\ell \;/\; C_0$  (95% CVL threshold)")
    ax.set_title(r"I and V detectability — noiseless (CVL)")
    ax.set_xticks(ell)

    box = dict(boxstyle="round", facecolor="white", alpha=0.9,
               linewidth=0.8, edgecolor="0.7")
    ax.text(0.97, 0.97, "Solid: $C_\\ell^I$\nDashed: $C_\\ell^V$",
            transform=ax.transAxes, fontsize=8, va="top", ha="right", bbox=box)

    plt.tight_layout()
    out = os.path.join(outdir, "CVL_combined.pdf")
    plt.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indir",  default=INDIR,  help="Directory with .npz data files.")
    parser.add_argument("--outdir", default=OUTDIR, help="Directory to save plots.")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Discover all CVL_Npsr*.npz files and build summary on the fly
    per_n   = discover_per_n(args.indir)
    summary = build_summary_from_per_n(per_n)
    n_arr   = summary["n_pulsars"]
    ell     = summary["ell"]

    print(f"Found {len(n_arr)} data file(s): N_pulsars = {n_arr}, ell = {ell}")

    # Load NANOGrav reference
    if os.path.exists(NG15_DAT):
        ng15 = np.loadtxt(NG15_DAT)[:, 1]
        # Trim/extend to match ell if needed
        ng15 = ng15[:len(ell)]
    else:
        print(f"Warning: {NG15_DAT} not found — NANOGrav reference omitted.")
        ng15 = None

    # Generate figures
    if ng15 is not None:
        fig_Cl_I(summary, per_n, ell, ng15, args.outdir)
    else:
        print("Skipping Fig 1 (no NANOGrav reference data).")

    fig_Cl_V(summary, per_n, ell, args.outdir)
    fig_ratio_vs_N(summary, per_n, ell, args.outdir)

    if ng15 is not None:
        fig_combined(summary, per_n, ell, ng15, args.outdir)

    print("\nDone.")


if __name__ == "__main__":
    main()
