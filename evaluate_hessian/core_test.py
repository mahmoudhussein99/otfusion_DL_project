import numpy as np

import torch 

from util_core import load_models, load_dataset, evaluate_hessian, dump_as_json, plot_loss_landscape, plot_ev_density

import time
import os


## >>> CONFIG -----
config = {
        'seed': 42,
        'architecture': "vgg11_nobias",
        'dataset': "Cifar10",
        'parent1_cp_path': "../cifar_models/pruning/pruned_parents/model_0.pruned.intial.checkpoint",
        'parent2_cp_path': "../cifar_models/pruning/pruned_parents/model_1.pruned.intial.checkpoint",
        'fusion_cp_path': "../cifar_models/pruning/geometric/fused.initial.checkpoint", #../cifar_models/non_pruning/fusion_retraining/best.checkpoint
        'device_id' : 0,
        'use_cuda' : False,
        'batch_size': 4,
        'num_batches': 2,
        'top_ev': 3,
        "lambdas": np.linspace(-0.5, 0.5, 21).astype(np.float32).tolist(),
        "compute_ev_density": True
    }
## <<<

def main():
    start_time = time.time()
    
    # Set seed
    torch.manual_seed(config['seed'])

    # Load Models
    print("\n----- Loading Models -----")
    parent1, parent2, fusion = load_models(config)

    # Dataset
    print("\n----- Loading Dataset -----") 
    test_loader = load_dataset(config)
    
    # Evaluate
    print("\n----- Evaluation -----")
    loss_landscape, eigenvalues, traces, eigenvalue_density = evaluate_hessian(config, [parent1, parent2, fusion], test_loader)
    
    path = f"./results/{config['architecture']}_{config['dataset']}" + time.strftime("_%m-%d_%H-%M")
    os.makedirs(path, exist_ok=True)
    config['time_taken[s]'] =  time.time() - start_time
    dump_as_json(path, "config", config)

    # Dump raw outputs
    print("\n----- Output Results -----")
    dump_path = os.path.join(path, "raw_dumps")
    os.makedirs(dump_path, exist_ok=True)
    
    dump_as_json(dump_path, "loss_landscape", loss_landscape)
    dump_as_json(dump_path, "eigenvalues", eigenvalues)
    dump_as_json(dump_path, "traces", traces)
    if (config["compute_ev_density"]):
        dump_as_json(dump_path, "eigenvalue_density", eigenvalue_density)

    # Plots
    plot_loss_landscape(config, loss_landscape, path)
    if (config["compute_ev_density"]):
        plot_ev_density(eigenvalue_density, path)

    return

if __name__ == "__main__":
    main()