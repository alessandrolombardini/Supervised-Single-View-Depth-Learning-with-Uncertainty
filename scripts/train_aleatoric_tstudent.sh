# Log script
if [ -f "/workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/log/aleatoric_tstudent.out" ]; then
    rm /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/log/aleatoric_tstudent.out
fi
# Log slurm
if [ -f "/workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/slurms/R-training_laplacian.out" ]; then
    rm /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/slurms/R-training_laplacian.out
fi
# Checkpoint
if [ -d "/workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/checkpoints/aleatoric_tstudent" ]; then
    rm -rf /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/checkpoints/aleatoric_tstudent
fi

cd  /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/src
pip3 install --upgrade pip
pip3 install -r requirements.txt
python3 main.py --uncertainty "aleatoric_tstudent"
