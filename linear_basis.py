# Import global libs
import sys
import time
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from multiprocessing import Pool, cpu_count
# from concurrent.futures import ProcessPoolExecutor
import jax
import jax.numpy as jnp
from functools import partial

sys.path.append("../")

# Import local libs
import examples_utils as eu
import fastPTA.utils as ut
from fastPTA.signals import get_signal_model
from fastPTA.Fisher_code import compute_fisher
from fastPTA.angular_decomposition import spherical_harmonics as sph

# %%

# Total observation time in years
T_obs_yrs = 16.03

# Number of frequencies used in the analysis
n_frequencies = 30

# Number of pulsars used in the analysis
n_pulsars = 68

# Maximum value of l used in the analysis
l_max = 6

# Define the signal model
signal_model = get_signal_model("power_law")

# The analysis assumes a power law template, specify here the input parameters
log_amplitude = -7.1995  # log amplitude
tilt = 2.0  # Tilt

# Number of catalog realizations to be generated
N_realizations = 30

# Whether you want to regenerate the fisher data
regenerate_fisher_data = False

# Whether you want to regenerate the sample data
regenerate_sample_data = False

# Name of the outfile, no need for the extension
# (will be stored in generated_data/)
outname = "NGlike"

# %%

# Confidence level for the Cls
limit_cl = 0.95

# Number of samples used for each realization
n_points = int(1e4)

# Number of catalog realizations used in the analysis.
# If larger than N_realizations taxes that as max
N_realizations_in_plot = 32

# Number of cores to be used for parallel computations
num_cores = 1#cpu_count()

# %%

# All multipoles here are set to zero
signal_lm = (
    1e-30 / np.sqrt(4 * np.pi) * np.ones(sph.get_n_coefficients_real(l_max))
)

# The monopole is set to 1 (with the right normalization)
signal_lm[0] = 1.0 / np.sqrt(4 * np.pi)

# The two parameters characterizing the signal in frequency
signal_parameters = np.array([log_amplitude, tilt])

# Len of the parameters used in the analysis
shape_params = len(signal_parameters)

# The monopole is not included in the fisher analysis since it's degenerate
means = np.concatenate((signal_parameters, signal_lm[1:]))

# The number of parameters used in the fisher analysis
n_params = len(means)

# %%

# Dictionary with the kwargs to generete the pulsar catalogs
generate_catalog_kwargs = {
    "n_pulsars": n_pulsars,
    "save_catalog": True,
    "use_ng_positions": True,
    **eu.EPTAlike,
}

# Dictionary with the kwargs to generate noise and orf tensors
get_tensors_kwargs = {
    "path_to_pulsar_catalog": "pulsar_configurations/EPTA68.txt",
    "add_curn": False,
    "regenerate_catalog": True,
    "anisotropies": True,
    "l_max": l_max,
}

# Dictionary with the kwargs to generate the fisher matrix
fisher_kwargs = {
    "T_obs_yrs": T_obs_yrs,
    "n_frequencies": n_frequencies,
    "signal_model": signal_model,
    "signal_parameters": signal_parameters,
    "signal_lm": signal_lm,
}
# %%

try:
    if regenerate_fisher_data:
        raise FileNotFoundError("Flag forces regeneration")

    fisher = np.load("generated_data/" + outname + ".npz")["all_fishers"]

except FileNotFoundError:
    fisher = np.zeros((N_realizations, n_params, n_params))

    for index in tqdm.tqdm(range(N_realizations)):
        res = compute_fisher(
            **fisher_kwargs,
            get_tensors_kwargs=get_tensors_kwargs,
            generate_catalog_kwargs=generate_catalog_kwargs,
        )[-1]

        # Delete the monopole since it's degenerate
        fisher[index] = np.delete(np.delete(res, 2, axis=-1), 2, axis=-2)

    np.savez("generated_data/" + outname + ".npz", all_fishers=fisher)
# %%



# try:
#     if regenerate_sample_data:
#         raise FileNotFoundError("Flag forces regeneration")

#     Cl_limits = np.loadtxt("generated_data/Cl_limits_" + outname + ".dat")
#     Cl_limits_prior = np.loadtxt(
#         "generated_data/Cl_limits_prior_" + outname + ".dat"
#     )

#     if len(Cl_limits) != N_realizations:
#         raise FileNotFoundError("Flag forces regeneration")

# except FileNotFoundError:

#     cov = ut.compute_inverse(fisher)

#     for j in range(min(N_realizations_in_plot, N_realizations)):
#         arg_grid = [
#             [means, cov[j]]
#             for j in range(min(N_realizations_in_plot, N_realizations))
#         ]

#     def get_Cl_wrapper(parameters):
#         return sph.get_Cl_limits(
#             *parameters,
#             shape_params,
#             n_points=n_points,
#             limit_cl=limit_cl,
#             max_iter=100,
#             prior=5.0 / (4.0 * np.pi),
#         )

#     start = time.time()
#     with Pool(num_cores) as p:
#         res = np.array(p.map(get_Cl_wrapper, arg_grid, chunksize=1))
#     end = time.time()
#     print("Time to compute Cl limts:", end - start)
    
    

#     Cl_limits = res[:, 0]
#     Cl_limits_prior = res[:, 1]

#     Cl_limits_prior = Cl_limits_prior[np.isfinite(Cl_limits_prior[:, 0])]
#     np.savetxt("generated_data/Cl_limits_" + outname + ".dat", Cl_limits)
#     np.savetxt(
#         "generated_data/Cl_limits_prior_" + outname + ".dat", Cl_limits_prior
#     )

# # Normalize with C0
# Cl_limits *= 4 * np.pi
# Cl_limits_prior *= 4 * np.pi
# %%

cov = ut.compute_inverse(fisher)

for j in range(min(N_realizations_in_plot, N_realizations)):
    arg_grid = [
        [means, cov[j]]
        for j in range(min(N_realizations_in_plot, N_realizations))
    ]

def get_Cl_wrapper(parameters):
    return sph.get_Cl_limits(
        *parameters,
        shape_params,
        n_points=n_points,
        limit_cl=limit_cl,
        max_iter=100,
        prior=5.0 / (4.0 * np.pi),
    )

# start = time.time()
# with Pool(num_cores) as p:
#     res = np.array(p.map(get_Cl_wrapper, arg_grid, chunksize=1))
# end = time.time()
# print("Time to compute Cl limts:", end - start)
res = []
start = time.time()
for args in arg_grid:
    res.append(get_Cl_wrapper(args))
res = np.array(res)
end = time.time()
print("Time to compute Cl limits:", end - start)



Cl_limits = res[:, 0]
Cl_limits_prior = res[:, 1]

Cl_limits_prior = Cl_limits_prior[np.isfinite(Cl_limits_prior[:, 0])]
np.savetxt("generated_data/Cl_limits_" + outname + ".dat", Cl_limits)
np.savetxt(
    "generated_data/Cl_limits_prior_" + outname + ".dat", Cl_limits_prior
)
# %%


# Normalize with C0
Cl_limits *= 4 * np.pi
Cl_limits_prior *= 4 * np.pi

# %%

# Compute the mean and the errors on the Cls without prior
mean_Cl_limits = np.mean(Cl_limits, axis=0)
err_low_Cl_limits = mean_Cl_limits - np.quantile(Cl_limits, 0.025, axis=0)
err_high_Cl_limits = np.quantile(Cl_limits, 0.975, axis=0) - mean_Cl_limits

# Compute the mean and the errors on the Cls with the prior
mean_Cl_limits_prior = np.mean(Cl_limits_prior, axis=0)
err_low_Cl_limits_prior = mean_Cl_limits_prior - np.quantile(Cl_limits_prior, 0.025, axis=0)
err_high_Cl_limits_prior = (
    np.quantile(Cl_limits_prior, 0.975, axis=0) - mean_Cl_limits_prior
)

# %%

data_prior = np.random.uniform(
    -5.0 / (4.0 * np.pi), 5.0 / (4.0 * np.pi), (n_points, len(means[1:]))
)

Cl_prior_values = sph.get_CL_from_real_clm(data_prior.T)[1:]

Cl_prior = np.quantile(Cl_prior_values, limit_cl, axis=-1)

Cl_prior *= 4 * np.pi

print("Just prior:")
print("C_l / C_0 =", Cl_prior)

# %%

limits_NG = np.loadtxt("examples/examples_paper_anisotropies/data_paper_2/limits_Cl_powerlaw_lin_ng15.dat")[:, 1]

# %%

# Use mathtext (no external LaTeX install needed)
plt.rc("text", usetex=False)
plt.rc("font", family="serif", size=11)

fig = plt.figure(figsize=(0.4 * 12.0, 0.4 * 11.0), dpi=150, edgecolor="white")
ax = fig.add_subplot(1, 1, 1)
ax.tick_params(axis="both", which="both", labelsize=11, direction="in", width=0.5)
ax.xaxis.set_ticks_position("both")
ax.yaxis.set_ticks_position("both")


ax.semilogy(
    np.arange(1, l_max + 1),
    limits_NG,
    label="NANOGrav limits",
    color="green",
    marker="x",
    lw=0,
)

ax.semilogy(
    np.arange(1, l_max + 1),
    Cl_prior,
    color="blue",
    label="Only prior",
)

ax.errorbar(
    np.arange(1, l_max + 1),
    mean_Cl_limits,
    yerr=(err_low_Cl_limits, err_high_Cl_limits),
    color="red",
    label="Fisher",
    marker=".",
)

ax.errorbar(
    np.arange(1, l_max + 1),
    mean_Cl_limits_prior,
    yerr=(err_low_Cl_limits_prior, err_high_Cl_limits_prior),
    color="grey",
    label="Fisher with prior",
    marker=".",
)

plt.legend(loc="lower right", fontsize=10, handlelength=1.5)

props = dict(
    boxstyle="round",
    facecolor="white",
    alpha=1,
    linewidth=1,
    edgecolor="0.8",
)

for axis in ["top", "bottom", "left", "right"]:
    ax.spines[axis].set_linewidth(0.5)

ax.text(
    1,
    14,
    "NANOGrav positions, $T_{\\mathrm{obs}} \\sim 15 \\, \\mathrm{yr}$",
    horizontalalignment="left",
    fontsize=10,
    verticalalignment="top",
    bbox=props,
    linespacing=1.4,
)

# Axis labels
ax.set_xlabel(r"$\ell$")
ax.set_ylabel(r"$C_\ell / C_0$")
plt.ylim(0.3e-1, 20)

plt.tight_layout()
plt.savefig("plots/Cl_sens_lim_ng_like.pdf")
plt.show()
    
    


# Cl_limits = []
# Cl_limits_prior = []

# for j in range(N_realizations_in_plot):
#     cl, cl_prior = sph.get_Cl_limits(
#         means, cov[j],
#         shape_params,
#         n_points=n_points,
#         limit_cl=limit_cl,
#         max_iter=100,
#         prior=5.0 / (4.0 * np.pi),
#     )
#     Cl_limits.append(cl)
#     Cl_limits_prior.append(cl_prior)

# Cl_limits = np.array(Cl_limits)
# Cl_limits_prior = np.array(Cl_limits_prior)