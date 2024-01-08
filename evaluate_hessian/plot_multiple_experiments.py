import os
import numpy as np
import json

from util_core import plot_loss_landscape, plot_ev_density

basepath = './results' # DO NOT CHANGE
experiments = [
    'exp_cifar10_vgg11_unpruned_retrained',
    'exp_cifar10_vgg11_structured_pruned_30_retrained',
    'exp_cifar10_vgg11_structured_pruned_50_retrained',
    'exp_cifar10_vgg11_structured_pruned_70_retrained',
    'exp_cifar10_vgg11_unstructured_pruned_30_retrained',
    'exp_cifar10_vgg11_unstructured_pruned_50_retrained',
    'exp_cifar10_vgg11_unstructured_pruned_70_retrained',
]

for exp in experiments:
    path = os.path.join(basepath, exp)

    print(f'EXPERIMENT: {exp[4:]}')
    print('* Loading Raw Dumps', end = ' ')
    
    config = json.load(open(f'{path}/config.json'))
    loss_landscape = json.load(open(f'{path}/raw_dumps/loss_landscape.json'))
    eigenvalues = json.load(open(f'{path}/raw_dumps/eigenvalues.json'))
    traces = json.load(open(f'{path}/raw_dumps/traces.json'))
    eigenvalue_density = json.load(open(f'{path}/raw_dumps/eigenvalue_density.json'))
    
    print('* Plotting Experiment', end = ' ')

    plot_path = os.path.join(path, "final_plots")
    os.makedirs(plot_path, exist_ok=True)
    plot_loss_landscape(config, loss_landscape, plot_path)
    plot_ev_density(eigenvalue_density, plot_path)

    print('DONE!')
