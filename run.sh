#!/bin/bash
# add the configs how to run the code here with module loads
#SBATCH --job-name IhopeThisWillWorkForMeSignedDavid
#SBATCH --account EU-25-40
#SBATCH --partition qgpu
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --ntasks-per-node 64
#SBATCH --time 4:00:00

ml CUDA/12.8.0
ml Python/3.11.3-GCCcore-12.3.0
ml libjpeg-turbo/2.1.5.1-GCCcore-12.3.0

source /home/davidciz/silver-truth/.venv/bin/activate

python qa.py ml-run
