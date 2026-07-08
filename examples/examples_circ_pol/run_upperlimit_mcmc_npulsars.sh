#!/usr/bin/env bash
set -euo pipefail

# Local runner for zero-anisotropy I+V upper-limit MCMC sweeps.
# Edit these defaults, or override them from the environment:
#
#   N_PULSARS_LIST="50 100 200" MCMC_STEPS=5000 ./run_upperlimit_mcmc_npulsars.sh

N_PULSARS_LIST=${N_PULSARS_LIST:-"50 100 200 500"}
L_MAX=${L_MAX:-4}
N_FREQUENCIES=${N_FREQUENCIES:-20}
N_WALKERS=${N_WALKERS:-24}
BURNIN_STEPS=${BURNIN_STEPS:-1000}
MCMC_STEPS=${MCMC_STEPS:-4000}
I_MAX=${I_MAX:-10}
SEED=${SEED:-0}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

for N_PULSARS in ${N_PULSARS_LIST}; do
    echo "============================================================"
    echo "Running zero-anisotropy I+V upper-limit MCMC:"
    echo "  n_pulsars=${N_PULSARS}"
    echo "  l_max=${L_MAX}, n_frequencies=${N_FREQUENCIES}"
    echo "============================================================"

    python run_upperlimit_mcmc.py \
        --n-pulsars "${N_PULSARS}" \
        --l-max "${L_MAX}" \
        --n-frequencies "${N_FREQUENCIES}" \
        --n-walkers "${N_WALKERS}" \
        --burnin-steps "${BURNIN_STEPS}" \
        --mcmc-steps "${MCMC_STEPS}" \
        --i-max "${I_MAX}" \
        --seed "${SEED}"
done
