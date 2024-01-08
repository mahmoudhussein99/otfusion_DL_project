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
	'exp_cifar10_vgg11_unstructured_pruned_30_retrained',
	'exp_cifar10_vgg11_unstructured_pruned_50_retrained',
	'exp_cifar10_vgg11_unstructured_pruned_70_retrained',
	'exp_cifar10_vgg11_structured_pruned_30_retrained',
	'exp_cifar10_vgg11_structured_pruned_50_retrained',
	'exp_cifar10_vgg11_structured_pruned_70_retrained',
]
# used for tables
table_suffix = 'cifar10_vgg11' 
experiment_names = [
	'C10 V11 unP',
    'C10 V11 uP30',
    'C10 V11 uP50',
    'C10 V11 uP70',
    'C10 V11 sP30',
    'C10 V11 sP50',
    'C10 V11 sP70',
]
model_names = [
    'Parent 1',
    'Parent 2',
    'Child',
    'Child (retrained)',
]
dict_of_dicts = {
    'configs': [],
    'eigenvalues': [],
    'traces': [],
}

for exp in experiments:
    path = os.path.join(basepath, exp)

    print(f'EXPERIMENT: {exp[4:]}')
    print('* Loading Raw Dumps')

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
        print('* Outputting Table', end = ' ')
        dict_of_dicts['configs'].append(config)

        # Trace Table
        traces_dict = {}
        for key_m, value_m in traces.items():
            trace = np.array(list(value_m.values()))
            trace_mean = str(round(trace.mean(axis=0), 2))
            trace_std = str(round(trace.std(axis=0), 2))
            traces_dict[key_m] = fr'{trace_mean} {{\scriptsize $\pm$ {trace_std}}}' if trace_mean != str(0.0) else 'n/a'
        dict_of_dicts['traces'].append(traces_dict)

        # Eigenvalue Table
        eigenvalues_dict = {}
        for key_m, value_m in eigenvalues.items():
            eigenvalues = np.array(list(value_m.values()))
            eigenvalues_mean = str(round(eigenvalues.mean(), 2)) # over batches and top k
            eigenvalues_std = str(round(eigenvalues.std(), 2)) # over batches and top k
            eigenvalues_dict[key_m] = fr'{eigenvalues_mean} {{\scriptsize $\pm$ {eigenvalues_std}}}'
        dict_of_dicts['eigenvalues'].append(eigenvalues_dict)

        print('DONE!')


if TABLES:
    # import json
    # print(json.dumps(dict_of_dicts['traces']))

    output_table(
        row_names=experiment_names,
        col_names=model_names,
        dicts=dict_of_dicts['traces'],
        caption='Trace caption.',
        label='mylabel1',
        path = os.path.join(basepath, f'table_traces_{table_suffix}.latex')
    )

    output_table(
        row_names=experiment_names,
        col_names=model_names,
        dicts=dict_of_dicts['eigenvalues'],
        caption='Eigenvalues caption.',
        label='mylabel2',
        path = os.path.join(basepath, f'table_eigenvalues_{table_suffix}.latex')
    )
