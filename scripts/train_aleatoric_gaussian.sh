# Log script
path = /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/log/aleatoric_gaussian.out
if [ -e "$path" ]; then
    rm -rf "$path"
# Log slurm
path = /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/slurms/aleatoric_gaussian.out
if [ -e "$path" ]; then
    rm -rf "$path"
# Checkpoint
path = /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/results/checkpoints/aleatoric_gaussian
if [ -e "$path" ]; then
    rm -rf "$path"


cd  /workspace/Supervised-Single-View-Depth-Learning-with-Uncertainty/src
pip3 install --upgrade pip
pip3 install -r requirements.txt
python3 main.py --uncertainty "aleatoric_gaussian"
