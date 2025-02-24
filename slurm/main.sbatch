#!/bin/bash
#SBATCH -J Test-lzp
#SBATCH -p A800
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --qos qmultiple

srun -n 1 -c 8 singularity exec --nv \
                                --bind $PWD:$PWD \
                                /hpai/aios3.0/private/user/s2022244206/pmanet/lzp-slurm-pmanet.sif \
                                python --version
 