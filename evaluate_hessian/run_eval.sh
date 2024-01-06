#!/bin/bash

#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --account=deep_learning
#SBATCH --output=logs.analysis_cifar10_vgg11_pruned_experiments.out

python3 analyse_multiple_experiments.py 
