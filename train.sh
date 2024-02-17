#!/bin/bash

echo "Training Gaussian"
sbatch sbatch_train_gaussian.sh &
sleep 5m

echo "Training Laplacian"
sbatch sbatch_train_laplacian.sh &
sleep 5m

echo "Training T-Student"
sbatch sbatch_train_tstudent.sh &
slee 5m 

echo "Training Normal"
sbatch sbatch_train_normal.sh &

