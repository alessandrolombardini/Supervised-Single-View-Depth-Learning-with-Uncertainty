# Log script
if [ -f "/workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/log/aleatoric_gaussian.out" ]; then
    rm /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/log/aleatoric_gaussian.out
fi
# Log slurm
if [ -f "/workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/slurms/R-training_gaussian.out" ]; then
    rm /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/slurms/R-training_gaussian.out
fi
# Checkpoint
if [ -d "/workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/checkpoints/aleatoric_gaussian" ]; then
    rm -rf /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/checkpoints/aleatoric_gaussian
fi

cd  /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/src
pip3 install --upgrade pip
pip3 install -r requirements.txt
python3 main.py --uncertainty "aleatoric_gaussian"
