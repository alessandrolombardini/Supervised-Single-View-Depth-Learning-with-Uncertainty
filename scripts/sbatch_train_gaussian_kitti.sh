#!/bin/bash
#SBATCH --job-name=training_gaussian_fashion_kitti
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=20G
#SBATCH --output=results/slurms/training_gaussian_fashion_kitti.out

srun --output=results/log/aleatoric_gaussian_fashion_kitti.out --container-mounts=/raid/ropert/alombard:/workspace --container-workdir=/workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty --container-image=nvcr.io#nvidia/pytorch:23.09-py3 sh scripts/train/train_aleatoric_gaussian_fashion_kitti.sh

