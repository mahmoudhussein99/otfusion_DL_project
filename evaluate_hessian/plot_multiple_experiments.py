import os
import numpy as np
import json

#  from util_core import plot_loss_landscape, plot_ev_density
from generate_table import output_table

PLOTS = False
TABLES = True

basepath = './results' # DO NOT CHANGE
# used for plots
experiments = [
    'exp_cifar10_vgg11_unpruned_retrained',
    'exp_cifar10_vgg11_structured_pruned_30_retrained',
    'exp_cifar10_vgg11_structured_pruned_50_retrained',
    'exp_cifar10_vgg11_structured_pruned_70_retrained',
    'exp_cifar10_vgg11_unstructured_pruned_30_retrained',
    'exp_cifar10_vgg11_unstructured_pruned_50_retrained',
    'exp_cifar10_vgg11_unstructured_pruned_70_retrained',
]
# used for tables
experiment_names = [
    'unpurned',
    's.pruned 30'
    's.pruned 50'
    's.pruned 70'
    'us.pruned 30'
    'us.pruned 50'
    'us.pruned 70'
]
model_names = [
    'parent 1',
    'parent 2',
    'fusion (init)',
    'fusion (retrained)',
]
dict_of_dicts = {
    'configs': [],
    'eigenvalues': [],
    'traces': [],
}

for exp in experiments:
    path = os.path.join(basepath, exp)

    print(f'EXPERIMENT: {exp[4:]}')
    print('* Loading Raw Dumps', end = ' ')

    config = json.load(open(f'{path}/config.json'))
    loss_landscape = json.load(open(f'{path}/raw_dumps/loss_landscape.json'))
    eigenvalues = json.load(open(f'{path}/raw_dumps/eigenvalues.json'))
    traces = json.load(open(f'{path}/raw_dumps/traces.json'))
    eigenvalue_density = json.load(open(f'{path}/raw_dumps/eigenvalue_density.json'))


    if PLOTS:
        print('* Plotting Experiment', end = ' ')

        plot_path = os.path.join(path, "final_plots")
        os.makedirs(plot_path, exist_ok=True)
        #  plot_loss_landscape(config, loss_landscape, plot_path)
        #  plot_ev_density(eigenvalue_density, plot_path)

        print('DONE!')

    if TABLES:
        dict_of_dicts['configs'].append(config)
        dict_of_dicts['eigenvalues'].append(eigenvalues)
        dict_of_dicts['traces'].append(traces)


if TABLES:

    output_table(
        row_names=experiment_names,
        col_names=model_names,
        dicts=dict_of_dicts['eigenvalues'],
        caption='Trace caption',
        label='mylabel',
        path = os.path.join(basepath, 'table_traces.latex')
    )
