#!/bin/bash

echo "Training Gaussian"
sbatch sbatch_train_gaussian.sh
echo "Training Laplacian"
sbatch sbatch_train_laplacian.sh
echo "Training T-Student"
sbatch sbatch_train_tstudent.sh