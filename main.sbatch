#!/bin/bash
#SBATCH -J Test-lzp
#SBATCH -p A800
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --qos qmultiple

srun -n 1 -c 16 singularity exec --nv \
                                --bind $PWD:$PWD \
                                slurm/lzp-slurm-pmanet.sif \
                                python3 Train.py
 