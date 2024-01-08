#!/bin/bash

#SBATCH --gpus=1
#SBATCH --time=30:00:00
#SBATCH --account=deep_learning
#SBATCH --output=logs.analysis_cifar100_vgg11_experiments.out

python3 analyse_multiple_experiments.py 
