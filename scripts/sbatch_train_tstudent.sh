#!/bin/bash
#SBATCH --job-name=training_tstudent
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=20G
#SBATCH --output=../results/slurms/R-%x.out

srun --output=../results/log/aleatoric_tstudent.out --container-mounts=/raid/ropert/alombard:/workspace --container-workdir=/workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty --container-image=nvcr.io#nvidia/pytorch:23.09-py3 sh train/train_aleatoric_tstudent.sh

