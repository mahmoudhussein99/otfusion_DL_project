#!/bin/bash

#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=logs.out

python3 ./core_test.py