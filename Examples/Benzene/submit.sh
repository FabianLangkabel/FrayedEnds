#!/bin/bash
#
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=htc
#SBATCH --mem=32G
#SBATCH --time=72:00:00

cd /workspace/MRA_Projects/projects_fl/Benzene/MW-FCI_run1

export MAMBA_ROOT_PREFIX=/workspace/MRA_Projects/micromamba
eval "$(/workspace/MRA_Projects/micromamba/bin/micromamba shell hook -s posix)"
micromamba activate MRA

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspace/MRA_Projects/anaconda3/envs/madness-oblas/lib
export MAD_NUM_THREADS=8
python run.py >> run_out.log
