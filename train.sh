#!/bin/bash

echo "Training Gaussian"  >> output.txt
sbatch scripts/sbatch_train_gaussian.sh &
sleep 300

echo "Training Laplacian"  >> output.txt
sbatch scripts/sbatch_train_laplacian.sh &
sleep 300

echo "Training T-Student"  >> output.txt
sbatch scripts/sbatch_train_tstudent.sh &
sleep 300

echo "Training Normal"  >> output.txt
sbatch scripts/sbatch_train_normal.sh &

