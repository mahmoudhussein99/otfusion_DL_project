#!/bin/sh
python3 main.py --gpu-id 0 --model-name resnet18_nobias_nobn --n-epochs 300 --save-result-file sample_cifar100_resnet18_nobias_nobn_no_pruning.csv --sweep-name exp_cifar100_resnet18_no_pruning --correction --ground-metric  euclidean --weight-stats --geom-ensemble-type acts --ground-metric-normalize none --sweep-id 90 --ckpt-type best --dataset Cifar100 --ground-metric-eff --recheck-cifar --activation-seed 21 --prelu-acts --past-correction --not-squared --exact --experiment-name cifar100_resnet18_nobias_nobn_no_pruning --retrain 300 --retrain-geometric-only --num-models 2 --to-download