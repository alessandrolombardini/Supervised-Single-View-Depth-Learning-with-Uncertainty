# Log script
#if [ -f "/workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/log/aleatoric.laplacian.out" ]; then
#    echo "Deleting results/log/aleatoric.laplacian.out..."
#    rm /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/log/aleatoric.laplacian.out
#fi
# Log slurm
#if [ -f "/workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/slurms/R-training_laplacian.out" ]; then
#    echo "Deleting results/slurms/T-training_laplacian.out..."
#    rm /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/slurms/R-training_laplacian.out
#fi
# Checkpoint
#if [ -d "/workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/checkpoints/aleatoric.laplacian" ]; then
#    echo "Deleting results/checkpoints/aleatoric.laplacian..."
#    rm -rf /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/checkpoints/aleatoric.laplacian
#fi

cd  /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/src
pip3 install --upgrade pip
pip3 install -r requirements.txt
python3 main.py --uncertainty "aleatoric.laplacian" --data_name "cifar10"
