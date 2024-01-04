import numpy as np
import pandas as pd

import torch 
from torchvision import datasets, transforms

import sys
sys.path.append("../")
import cifar.models

from pyhessian_mod import hessian # Hessian computation
from density_plot import density_generate

import matplotlib.pyplot as plt

import copy
import os
import time
import json

def get_model(config):
    num_classes = 100 if config['dataset'] == 'Cifar100' else 10
    
    relu_inplace=True

    model = {
        'vgg11_nobias': lambda: cifar.models.VGG('VGG11', num_classes, batch_norm=False, bias=False, relu_inplace=relu_inplace),
        'vgg11_half_nobias': lambda: cifar.models.VGG('VGG11_half', num_classes, batch_norm=False, bias=False,
                                           relu_inplace=relu_inplace),
        'vgg11_doub_nobias': lambda: cifar.models.VGG('VGG11_doub', num_classes, batch_norm=False, bias=False,
                                           relu_inplace=relu_inplace),
        'vgg11_quad_nobias': lambda: cifar.models.VGG('VGG11_quad', num_classes, batch_norm=False, bias=False,
                                           relu_inplace=relu_inplace),
        'vgg11':     lambda: cifar.models.VGG('VGG11', num_classes, batch_norm=False, relu_inplace=relu_inplace),
        'vgg11_bn':  lambda: cifar.models.VGG('VGG11', num_classes, batch_norm=True, relu_inplace=relu_inplace),
        'resnet18':  lambda: cifar.models.ResNet18(num_classes=num_classes),
        'resnet18_nobias': lambda: cifar.models.ResNet18(num_classes=num_classes, linear_bias=False),
        'resnet18_nobias_nobn': lambda: cifar.models.ResNet18(num_classes=num_classes, use_batchnorm=False, linear_bias=False),
    }[config['architecture']]()
    return model

def load_models(config):
    parent1 = get_model(config)
    parent2 = get_model(config)
    fusion = get_model(config)

    device = 'cpu'

    if(config['use_cuda']):
        parent1.cuda(config['device_id'])
        parent2.cuda(config['device_id'])
        fusion.cuda(config['device_id'])

        device = 'cuda:' + str(config['device_id'])

    state1 = torch.load(config['parent1_cp_path'], map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, device)
            ))
    state2 = torch.load(config['parent2_cp_path'], map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, device)
            ))
    state3 = torch.load(config['fusion_cp_path'], map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, device)
            ))

    parent1.load_state_dict(state1['model_state_dict'])
    parent2.load_state_dict(state2['model_state_dict'])
    fusion.load_state_dict(state3['model_state_dict'])
    
    return parent1, parent2, fusion

def load_dataset(config):
    if(config['dataset'] == 'Cifar100'):
        test_dataset = datasets.CIFAR100('./data/', train=False, download=True,
                                transform=transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.Normalize(
                                        # (mean_ch1, mean_ch2, mean_ch3), (std1, std2, std3)
                                        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ]))
    else:
        test_dataset = datasets.CIFAR10('./data/', train=False, download=True,
                                    transform=transforms.Compose([
                                            transforms.ToTensor(),
                                            transforms.Normalize(
                                            # (mean_ch1, mean_ch2, mean_ch3), (std1, std2, std3)
                                            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                    ]))

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = config['batch_size'], shuffle=True)
    return test_loader

def get_params(model,  model_perturbed, direction, alpha):
    for m, m_perturbed, d in zip(model.parameters(), model_perturbed.parameters(), direction):
        m_perturbed.data = m.data + alpha * d
    return model_perturbed

def evaluate_hessian(config, models, loader):
    criterion = torch.nn.CrossEntropyLoss()

    ev_total = []
    trace_total = []
    loss_list_total = []
    ev_density_total = []

    for k, (inputs, targets) in enumerate(loader):
        if config['num_batches'] > 0 and k >= config['num_batches']: break
        
        if(config['use_cuda']):
            inputs, targets = inputs.cuda(config['device_id']), targets.cuda(config['device_id'])

        ev_models = []
        trace_models = []
        loss_list_models = []
        ev_density_models = []
        
        for i, model in enumerate(models):
            hessian_comp = hessian(model, criterion, data=(inputs, targets), cuda = config['use_cuda'])
            trace = hessian_comp.trace()

            # EV Computation
            top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n = config['top_ev'])

            # Density Computation
            if (config["compute_ev_density"]):
                density_eigen, density_weight = hessian_comp.density()
                ev_density_models.append([density_eigen, density_weight])

            loss_list = []
            
            # create a copy of the model
            model_perburbed = copy.deepcopy(model)
            model_perburbed.eval()
            
            if config['use_cuda']:
                model_perburbed = model_perburbed.cuda(config['device_id'])

            for lamda in config['lambdas']:
                # Perturb by eigenvector corresponding with largest eigenvalue!
                model_perburbed = get_params(model, model_perburbed, top_eigenvector[0], lamda)
                loss_list.append(criterion(model_perburbed(inputs), targets).item())
            
            ev_models.append(top_eigenvalues)
            loss_list_models.append(loss_list)
            trace_models.append(np.mean(trace))
        
        ev_total.append(ev_models)
        loss_list_total.append(loss_list_models)
        trace_total.append(trace_models)
        ev_density_total.append(ev_density_models)

    return loss_list_total, ev_total, trace_total, ev_density_total

def output_results(config, loss, evs, traces, ev_density):

    path = f"./results/{config['architecture']}_{config['dataset']}" + time.strftime("_%m-%d_%H-%M")
    os.makedirs(path, exist_ok=True)

    config['time_taken[s]'] =  time.time() - config['time_taken[s]']
    config['lambdas'] = config['lambdas'].tolist()
 
    with open(f"{path}/config.json", "w") as outfile:
        outfile.write(json.dumps(config, indent=4))

    # Traces of models
    traces_np = np.array(traces)
    mean_traces = traces_np.mean(axis=0)
    std_traces = traces_np.std(axis=0)

    # Top EigenValues of models
    evs_np = np.array(evs)
    mean_evs = evs_np.mean(axis=0)
    std_evs = evs_np.std(axis=0)


    # Store as .csv
    trace_df = pd.DataFrame({"Model" : ["Parent 1", "Parent 2", "Fusion"],
                            "Trace mean": mean_traces, 
                             "Trace std": std_traces})

    ev_df = pd.DataFrame({"Parent 1": mean_evs[0], 
                            "Parent 2" : mean_evs[1], 
                            "Fusion": mean_evs[2]})

    trace_df.to_csv(f"{path}/trace.csv", index=False)
    ev_df.to_csv(f"{path}/ev.csv", index=False)

    # Loss Landscape Perturbed by Top EV of Hessian
    loss_np = np.array(loss)
    mean_loss = loss_np.mean(axis=0)
    std_loss = loss_np.std(axis=0)

    labels = ["parent1", "parent2", "fusion"]
    colors = ["blue", "orange", "red"]

    for i in range(mean_loss.shape[0]):
        plt.fill_between(config['lambdas'], mean_loss[i] + std_loss[i], mean_loss[i] - std_loss[i], alpha = 0.25, color = colors[i])
        plt.plot(config['lambdas'], mean_loss[i], label=labels[i], color = colors[i])

    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Perturbation')
    plt.title('Loss landscape perturbed based on top Hessian eigenvector')
    plt.savefig(fname=f"{path}/hessian_ev_perturb.png")

    # Density of EVs for each model and batch
    if (config["compute_ev_density"]):
        fig, axs = plt.subplots(len(ev_density), len(ev_density[0]))

        for i, batch in enumerate(ev_density):
            for j, [density_eigen, density_weight] in enumerate(batch):
                density, grids = density_generate(density_eigen, density_weight)
                axs[i, j].semilogy(density, grids + 1.0e-7, label=labels[j], color = colors[j])
                axs[i, j].set_title(f"{labels[j]} - Batch {i}")
                axs[i, j].set(xlim=(np.min(density_eigen) - 1, np.max(density_eigen) + 1), ylim=(None, None))
                # plt.ylabel('Density (Log Scale)', fontsize=14, labelpad=10)
                # aplt.xlabel('Eigenvlaue', fontsize=14, labelpad=10)
                # plt.xticks(fontsize=12)
                # plt.yticks(fontsize=12)
                # plt.axis([np.min(eigenvalues) - 1, np.max(eigenvalues) + 1, None, None])
                # plt.tight_layout()
        fig.tight_layout()
        fig.savefig(f'{path}/ev_density_plot.png')