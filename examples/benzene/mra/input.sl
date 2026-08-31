#!/usr/bin/env bash
 
#SBATCH --job-name=input
#SBATCH --partition=epyc
#SBATCH --mem=64G
#SBATCH --mail-type=END,INVALID_DEPEND
#SBATCH --mail-user=thuy.truong@uni-a.de
#SBATCH --output=logs/slurm-%j.out
#SBATCH --time=7-0
#SBATCH --cpus-per-task=28

module load intel/2024.2.1
module load anaconda
conda activate fe_paper

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MAD_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export LD_PRELOAD=/hpc/gpfs2/scratch/u/truonthu/.conda/envs/fe_paper/lib/python3.12/site-packages/block2.libs/libfixcpu.so

srun --unbuffered python input.py