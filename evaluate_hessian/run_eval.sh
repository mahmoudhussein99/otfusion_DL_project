#!/bin/bash

#SBATCH --gpus=1
#SBATCH --time=01:00:00
#SBATCH --output=logs.out

python3 ./core_test.py