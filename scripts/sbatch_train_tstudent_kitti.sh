#!/bin/bash
#SBATCH --job-name=tstudent_kitti
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=20G
#SBATCH --output=results/slurms/tstudent_kitti.out

srun --output=results/log/tstudent_kitti.out --container-mounts=/raid/ropert/alombard:/workspace --container-workdir=/workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty --container-image=nvcr.io#nvidia/pytorch:23.09-py3 sh scripts/train/train_aleatoric_tstudent_kitti.sh

