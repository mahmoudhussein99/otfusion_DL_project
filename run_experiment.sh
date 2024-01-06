#!/bin/bash

#SBATCH --gpus=1
#SBATCH --time=05:00:00
#SBATCH --output=logs.cifar10_resnet18_nobias_nobn_structured_pruning_70.out

# Reset18 - Cifar10 - No Pruning
# python3 main.py --gpu-id 0 --model-name resnet18_nobias_nobn --n-epochs 300 --save-result-file sample_cifar10_resnet18_nobias_nobn_no_pruning.csv \
# --sweep-name exp_cifar10_resnet18_nobias_nobn_no_pruning --correction --ground-metric  euclidean --weight-stats \
# --activation-histograms --activation-mode raw --geom-ensemble-type acts --act-num-samples 200 --ground-metric-normalize none --sweep-id 90 --ckpt-type best --to-download --dataset Cifar10 \
# --ground-metric-eff --recheck-cifar --recheck-acc --activation-seed 21 --prelu-acts --past-correction --not-squared \
# --exact --experiment-name cifar10_resnet18_nobias_nobn_no_pruning --retrain 300 --retrain-geometric-only --load-models ./resnet_models/ --handle-skips

# Reset18 - Cifar10 - Pruning
python3 main.py --gpu-id 0 --model-name resnet18_nobias_nobn --n-epochs 300 --save-result-file sample_cifar10_resnet18_nobias_nobn_structured_pruning_70.csv \
--sweep-name exp_cifar10_resnet18_nobias_nobn_structured_pruning_70 --correction --ground-metric  euclidean --weight-stats \
--activation-histograms --activation-mode raw --geom-ensemble-type acts --act-num-samples 200 --ground-metric-normalize none --sweep-id 90 --ckpt-type best --to-download --dataset Cifar10 \
--ground-metric-eff --recheck-cifar --recheck-acc --activation-seed 21 --prelu-acts --past-correction --not-squared \
--exact --experiment-name cifar10_resnet18_nobias_nobn_structured_pruning_70 --retrain 300 --retrain-geometric-only --load-models ./resnet_models/ --handle-skips \
--prune --prune-frac 0.7 --prune-type structured

# python3 main.py --gpu-id 0 --model-name resnet18_nobias_nobn --n-epochs 300 --save-result-file sample.csv \
# --sweep-name exp_sample --exact --correction --ground-metric euclidean --weight-stats \
# --activation-histograms --activation-mode raw --geom-ensemble-type acts --act-num-samples 200 --sweep-id 21 \
# --ground-metric-normalize none --activation-seed 21 --prelu-acts --recheck-acc \
# --load-models ./resnet_models/ --ckpt-type best --past-correction --not-squared  --dataset Cifar10 \
# --handle-skips

# python3 main.py --gpu-id -1 --model-name resnet18_nobias_nobn --n-epochs 300 --save-result-file sample_cifar10_resnet18_nobias_nobn_structured_pruning_70.csv \
# --sweep-name exp_cifar10_resnet18_nobias_nobn_structured_pruning_70 --correction --ground-metric  euclidean --weight-stats \
# --geom-ensemble-type wts --ground-metric-normalize none --sweep-id 90 --ckpt-type best --to-download --dataset Cifar10 \
# --ground-metric-eff --recheck-cifar --recheck-acc --activation-seed 21 --prelu-acts --past-correction --not-squared \
# --exact --experiment-name cifar10_resnet18_nobias_nobn_structured_pruning_70 --retrain 300 --retrain-geometric-only --load-models ./resnet_models/ --handle-skips