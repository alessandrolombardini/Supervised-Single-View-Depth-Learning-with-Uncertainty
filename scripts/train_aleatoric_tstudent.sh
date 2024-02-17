# Log script
path = /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/log/aleatoric_tstudent.out
if [ -f "$path" ]; then
    rm /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/log/aleatoric_tstudent.out
# Log slurm
path = /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/slurms/R-training_laplacian.out
if [ -f "$path" ]; then
    rm /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/slurms/R-training_laplacian.out
# Checkpoint
path = /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/checkpoints/aleatoric_tstudent
if [ -d "$path" ]; then
    rm -rf /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/checkpoints/aleatoric_tstudent

cd  /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/src
pip3 install --upgrade pip
pip3 install -r requirements.txt
python3 main.py --uncertainty "aleatoric_tstudent"
