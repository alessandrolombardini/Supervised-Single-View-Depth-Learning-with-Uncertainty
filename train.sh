#!/bin/bash

echo "Training Normal - KITTI"  >> output.txt
sbatch scripts/sbatch_train_normal_kitti.sh &
sleep 300
echo "Training Gaussian - KITTI"  >> output.txt
sbatch scripts/sbatch_train_gaussian_kitti.sh &
sleep 300
echo "Training T-Student - KITTI"  >> output.txt
sbatch scripts/sbatch_train_tstudent_kitti.sh &
sleep 300
echo "Training Laplacian - KITTI"  >> output.txt
sbatch scripts/sbatch_train_laplacian_kitti.sh &
sleep 300
