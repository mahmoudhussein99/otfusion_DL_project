import os
import numpy as np
from core_test import analyse_hessian

basepath = '../../models_for_evaluation' # DO NOT CHANGE
experiments = [
    'exp_cifar10_vgg11_unpruned_retrained',
    'exp_cifar10_vgg11_structured_pruned_30_retrained',
    'exp_cifar10_vgg11_structured_pruned_50_retrained',
    'exp_cifar10_vgg11_structured_pruned_70_retrained',
    'exp_cifar10_vgg11_unstructured_pruned_30_retrained',
    'exp_cifar10_vgg11_unstructured_pruned_50_retrained',
    'exp_cifar10_vgg11_unstructured_pruned_70_retrained',
]
config = {
    'seed': 21, # used for torch and statistical computation of trace
    # DO NOT SET MANUALLY
    'experiment_name': '',
    'architecture': 'vgg11_nobias', # TODO: change me
    'dataset': 'Cifar10', # TODO: change me
    # DO NOT SET MANUALLY
    'parent1_cp_path': '',
    # DO NOT SET MANUALLY
    'parent2_cp_path': '',
    # DO NOT SET MANUALLY
    'fusion_initial_cp_path': '',
    # DO NOT SET MANUALLY
    'fusion_retrained_cp_path': '',
    'device_id' : 0,
    'use_cuda' : True,
    'batch_size': 1000, # 1000 = value as in original code of sidak pal
    'num_batches': -1,  # -1 = all batches in training set
    'top_ev': 3, # quickly consumes much time (for higher values)
    'lambdas': np.linspace(-0.5, 0.5, 21).astype(np.float32).tolist(),
    'compute_ev_density': True # set to True for plotting eigenvalue density
}

for exp in experiments:
    path = os.path.join(basepath, exp)

    config['experiment_name'] = exp
    config['parent1_cp_path'] = f"{path}/pruned_parents/model_0.pruned.initial.checkpoint"
    config['parent2_cp_path'] = f"{path}/pruned_parents/model_1.pruned.initial.checkpoint"
    config['fusion_initial_cp_path'] = f"{path}/geometric/fused.initial.checkpoint"
    config['fusion_retrained_cp_path'] = f"{path}/geometric/fused.initial.checkpoint"

    print(f'EXPERIMENT: {exp[4:]}')
    print('* analysing hessian', end = ' ')
    analyse_hessian(config)
    print('DONE!')
