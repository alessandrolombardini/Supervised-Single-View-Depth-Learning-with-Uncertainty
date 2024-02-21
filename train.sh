#!/bin/bash

echo "Training Gaussian - CIFAR10"  >> output.txt
sbatch scripts/sbatch_train_gaussian_cifar10.sh &
sleep 300
echo "Training Gaussian - MNIST"  >> output.txt
sbatch scripts/sbatch_train_gaussian_mnist.sh &
sleep 300
echo "Training Gaussian - FASHION MNIST"  >> output.txt
sbatch scripts/sbatch_train_gaussian_fashion_mnist.sh &
sleep 300


echo "Training Laplacian - CIFAR10"  >> output.txt
sbatch scripts/sbatch_train_laplacian_cifar10.sh &
sleep 300
echo "Training Laplacian - MNIST"  >> output.txt
sbatch scripts/sbatch_train_laplacian_mnist.sh &
sleep 300
echo "Training Laplacian - MNIST"  >> output.txt
sbatch scripts/sbatch_train_laplacian_fashion_mnist.sh &
sleep 300


echo "Training T-Student - CIFAR10"  >> output.txt
sbatch scripts/sbatch_train_tstudent_cifar10.sh &
sleep 300
echo "Training T-Student - MNIST"  >> output.txt
sbatch scripts/sbatch_train_tstudent_mnist.sh &
sleep 300
echo "Training T-Student - MNIST"  >> output.txt
sbatch scripts/sbatch_train_tstudent_fashion_mnist.sh &
sleep 300

echo "Training Normal - CIFAR10"  >> output.txt
sbatch scripts/sbatch_train_normal_cifar10.sh &
sleep 300
echo "Training Normal"  >> output.txt
sbatch scripts/sbatch_train_normal_mnist.sh &
sleep 300
echo "Training Normal"  >> output.txt
sbatch scripts/sbatch_train_normal_fashion_mnist.sh &

