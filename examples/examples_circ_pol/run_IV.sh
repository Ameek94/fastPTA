#!/bin/bash --login
###
###
#job name
#SBATCH --job-name=iv_fastpta
#job stdout file
#SBATCH --output=iv_fastpta.out
#job stderr file
#SBATCH --error=iv_fastpta.err
#maximum job time in D-HH:MM
#SBATCH --time=2-23:59
#SBATCH --account=scw2169
#SBATCH --exclude=scs[0092,0002,0004,0018]
#number of parallel processes (tasks) you are requesting - maps to MPI processes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#memory per process in MB
#SBATCH --mem-per-cpu=4096
#SBATCH --mail-type=END,FAIL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=ameek.malhotra@swansea.ac.uk     # Where to send mail


module load anaconda/2023.09
module load compiler/gnu/12/1.0
module load mpi/openmpi/4.1.5

source activate /scratch/s.ameek.malhotra/ptagw

export OMP_NUM_THREADS=12

# Map array index to python script

srun --mpi=pmix \
     --ntasks=$SLURM_NTASKS \
     --cpus-per-task=$SLURM_CPUS_PER_TASK \
     python run_injection_l1_IV.py --n-pulsars 50 --mcmc-steps 20000

srun --mpi=pmix \
     --ntasks=$SLURM_NTASKS \
     --cpus-per-task=$SLURM_CPUS_PER_TASK \
     python run_injection_l1_IV.py --n-pulsars 100 --mcmc-steps 20000

srun --mpi=pmix \
     --ntasks=$SLURM_NTASKS \
     --cpus-per-task=$SLURM_CPUS_PER_TASK \
     python run_injection_l1_IV.py --n-pulsars 200 --mcmc-steps 20000

srun --mpi=pmix \
     --ntasks=$SLURM_NTASKS \
     --cpus-per-task=$SLURM_CPUS_PER_TASK \
     python run_injection_l1_IV.py --n-pulsars 500 --mcmc-steps 20000