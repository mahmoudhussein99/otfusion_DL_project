import numpy as np

import torch 

from util_core import load_models, load_dataset, evaluate_hessian, output_results

import matplotlib.pyplot as plt

import time


## >>> CONFIG -----
CONFIG = {
        'seed': 21,
        'architecture': "vgg11_nobias",
        'dataset': "Cifar10",
        'parent1_cp_path': "../cifar_models/model_0/best.checkpoint",
        'parent2_cp_path': "../cifar_models/model_1/best.checkpoint",
        'fusion_cp_path': "../cifar_models/fusion_retraining/best.checkpoint",
        'device_id' : 0,
        'use_cuda' : True,
        'time_taken[s]': time.time(),
        'batch_size': 16,
        'num_batches': 2,
        'top_ev': 3,
        "lambdas": np.linspace(-0.5, 0.5, 21).astype(np.float32)
    }
## <<<

def main():
    # Set seed
    torch.manual_seed(CONFIG['seed'])

    # Load Models
    parent1, parent2, fusion = load_models(CONFIG)

    # Dataset 
    test_loader = load_dataset(CONFIG)
    
    # Evaluate
    result, evs, traces = evaluate_hessian(CONFIG, [parent1, parent2, fusion], test_loader)
    
    # Output
    output_results(CONFIG, result, evs, traces)

    return

if __name__ == "__main__":
    main()