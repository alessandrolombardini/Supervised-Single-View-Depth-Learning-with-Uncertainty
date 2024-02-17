# Log script
path = /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/log/aleatoric_tstudent.out
if [ -e "$path" ]; then
    rm -rf "$path"
# Log slurm
path = /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/slurms/R-training_laplacian.out
if [ -e "$path" ]; then
    rm -rf "$path"
# Checkpoint
path = /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/checkpoints/aleatoric_tstudent
if [ -e "$path" ]; then
    rm -rf "$path"cd  /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/src

pip3 install --upgrade pip
pip3 install -r requirements.txt
python3 main.py --uncertainty "aleatoric_tstudent"
