#!/bin/bash

#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --output=logs.analysis_pruned_experiments_DEBUG.out

python3 analyse_multiple_experiments.py 
