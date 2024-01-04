#!/bin/bash

#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --output=logs_pruned.out


python3 main.py --gpu-id 0 --model-name vgg11_nobias --n-epochs 300 --save-result-file sample_pruned.csv \
--sweep-name exp_sample_pruned --correction --ground-metric euclidean --weight-stats \
--geom-ensemble-type wts --ground-metric-normalize none --sweep-id 90 --load-models ./cifar_models/ \
--ckpt-type best --dataset Cifar10 --ground-metric-eff --recheck-cifar --activation-seed 21 \
--prelu-acts --past-correction --not-squared --exact