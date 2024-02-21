# Log script
#if [ -f "/workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/log/aleatoric.gaussian.out" ]; then
#    echo "Deleting results/log/aleatoric.gaussian.out..."
#    rm /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/log/aleatoric.gaussian.out
#fi
# Log slurm
#if [ -f "/workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/slurms/R-training_gaussian.out" ]; then
#    echo "Deleting results/slurms/T-training_gaussian.out..."
#    rm /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/slurms/R-training_gaussian.out
#fi
# Checkpoint
#if [ -d "/workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/checkpoints/aleatoric.gaussian" ]; then
#    echo "Deleting results/checkpoints/aleatoric.gaussian..."
#    rm -rf /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/checkpoints/aleatoric.gaussian
#fi

cd  /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/src
pip3 install --upgrade pip
pip3 install -r requirements.txt
python3 main.py --uncertainty "aleatoric.gaussian"  --data_name "mnist"
