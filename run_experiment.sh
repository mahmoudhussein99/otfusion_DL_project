#!/bin/bash

#SBATCH --gpus=1
#SBATCH --time=05:00:00
#SBATCH --output=cifar10_resnet18_nobias_nobn_no_pruning.out

python3 main.py --gpu-id 0 --model-name resnet18_nobias_nobn --n-epochs 300 --save-result-file sample_cifar10_resnet18_nobias_nobn_no_pruning.csv \
--sweep-name exp_cifar10_resnet18_nobias_nobn_no_pruning --correction --ground-metric  euclidean --weight-stats \
--geom-ensemble-type wts --ground-metric-normalize none --sweep-id 90 --ckpt-type best --to-download --dataset Cifar10 \
--ground-metric-eff --recheck-cifar --activation-seed 21 --prelu-acts --past-correction --not-squared \
--exact --experiment-name cifar10_resnet18_nobias_nobn_no_pruning --retrain 300 --retrain-geometric-only --load-models ./cifar_models/