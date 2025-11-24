#!/bin/bash
# add the configs how to run the code here with module loads
#SBATCH --job-name Test_run_alpha1_JC
#SBATCH --account EU-25-40
#SBATCH --partition qgpu
#SBATCH --nodes 1
#SBATCH --gpus 1
#SBATCH --ntasks-per-node 64
#SBATCH --time 0:05:00

ml CUDA/12.8.0
ml Python/3.11.3-GCCcore-12.3.0

python -m venv .venv
source .venv/bin/activate

python cli_ensemble.py ensemble-experiment --name "test_ensemble_experiment_1" --parquet_file "data/ensemble_data/datasets/v1.00/ensemble_dataset_v1.00_split70-15-15_seed42.parquet"