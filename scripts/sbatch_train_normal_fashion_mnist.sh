#!/bin/bash
#SBATCH --job-name=training_normal_fashion_mnist
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=20G
#SBATCH --output=results/slurms/training_normal_fashion_mnist.out

srun --output=results/log/normal_fashion_mnist.out --container-mounts=/raid/ropert/alombard:/workspace --container-workdir=/workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty --container-image=nvcr.io#nvidia/pytorch:23.09-py3 sh scripts/train/training_normal_fashion_mnist.sh

