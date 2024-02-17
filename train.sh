#!/bin/bash

echo "Training Gaussian"  >> output.txt
sbatch sbatch_train_gaussian.sh &
sleep 5m

echo "Training Laplacian"  >> output.txt
sbatch sbatch_train_laplacian.sh &
sleep 5m

echo "Training T-Student"  >> output.txt
sbatch sbatch_train_tstudent.sh &
slee 5m 

echo "Training Normal"  >> output.txt
sbatch sbatch_train_normal.sh &

