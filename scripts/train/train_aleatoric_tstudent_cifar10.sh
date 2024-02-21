# Log script
#if [ -f "/workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/log/aleatoric.tstudent.out" ]; then
#    echo "Deleting results/log/aleatoric.tstudent.out..."
#    rm /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/log/aleatoric.tstudent.out
#fi
# Log slurm
#if [ -f "/workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/slurms/R-training.tstudent.out" ]; then
#    echo "Deleting results/slurms/T-training_tstudent.out..."
#    rm /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/slurms/R-training.tstudent.out
#fi
# Checkpoint
#if [ -d "/workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/checkpoints/aleatoric.tstudent" ]; then
#    echo "Deleting results/checkpoints/aleatoric.tstudent..."
#    rm -rf /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/checkpoints/aleatoric.tstudent
#fi

cd  /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/src
pip3 install --upgrade pip
pip3 install -r requirements.txt
python3 main.py --uncertainty "aleatoric.tstudent" --data_name "cifar10"
