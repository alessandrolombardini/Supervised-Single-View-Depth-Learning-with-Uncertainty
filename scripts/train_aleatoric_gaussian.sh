# Log script
path = /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/log/aleatoric_gaussian.out
if [ -f "$path" ]; then
    rm /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/log/aleatoric_gaussian.out
# Log slurm
path = /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/slurms/R-training_gaussian.out
if [ -f "$path" ]; then
    rm /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/slurms/R-training_gaussian.out
# Checkpoint
path = /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/checkpoints/aleatoric_gaussian
if [ -d "$path" ]; then
    rm -rf /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/checkpoints/aleatoric_gaussian


cd  /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/src
pip3 install --upgrade pip
pip3 install -r requirements.txt
python3 main.py --uncertainty "aleatoric_gaussian"
