import numpy as np

import torch
from torchvision import datasets, transforms

import sys
sys.path.append("../")
import cifar.models

from pyhessian_mod import hessian # Hessian computation
from density_plot import density_generate

import matplotlib.pyplot as plt

import copy
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
    fusion_initial = get_model(config)
    fusion_retrained = get_model(config)

    device = 'cpu'

    if(config['use_cuda']):
        parent1.cuda(config['device_id'])
        parent2.cuda(config['device_id'])
        fusion_initial.cuda(config['device_id'])
        fusion_retrained.cuda(config['device_id'])

        device = 'cuda:' + str(config['device_id'])

    state1 = torch.load(config['parent1_cp_path'], map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, device)
            ))
    state2 = torch.load(config['parent2_cp_path'], map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, device)
            ))
    state3 = torch.load(config['fusion_initial_cp_path'], map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, device)
            ))
    state4 = torch.load(config['fusion_retrained_cp_path'], map_location=(
                lambda s, _: torch.serialization.default_restore_location(s, device)
            ))

    parent1.load_state_dict(state1['model_state_dict'])
    parent2.load_state_dict(state2['model_state_dict'])
    fusion_initial.load_state_dict(state3['model_state_dict'])
    fusion_retrained.load_state_dict(state4['model_state_dict'])

    return parent1, parent2, fusion_initial, fusion_retrained

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
    # labels for dumping dict
    labels = ["parent1", "parent2", "fusion_initial", "fusion_retrained"]

    eigenvalues = {}
    traces = {}
    loss_landscape = {'lambdas': config['lambdas']}

    eigenvalue_density = {}

    for i, model in enumerate(models):
        eigenvalues[f"{labels[i]}"] = {}
        traces[f"{labels[i]}"] = {}
        loss_landscape[f"{labels[i]}"] = {}
        eigenvalue_density[f"{labels[i]}"] = {}

        for k, (inputs, targets) in enumerate(loader):
            if config['num_batches'] > 0 and k >= config['num_batches']: break

            if(config['use_cuda']):
                inputs, targets = inputs.cuda(config['device_id']), targets.cuda(config['device_id'])

            # Hessian
            hessian_comp = hessian(model, criterion, data=(inputs, targets), cuda = config['use_cuda'])

            # Trace Computation
            trace = hessian_comp.trace()

            # Eigenvalue Computation
            top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n = config['top_ev'])

            # Eigenvalue Density Computation
            if (config["compute_ev_density"]):
                density_eigen, density_weight = hessian_comp.density()
                eigenvalue_density[f"{labels[i]}"][f"batch_{k}"] = [density_eigen, density_weight]

            # Loss Landscape Perturbation Computation
            loss_list = []

            # create a copy of the model
            model_perburbed = copy.deepcopy(model)
            model_perburbed.eval()

            if config['use_cuda']:
                model_perburbed = model_perburbed.cuda(config['device_id'])

            for lam in config['lambdas']:
                # Perturb by eigenvector corresponding with largest eigenvalue!
                model_perburbed = get_params(model, model_perburbed, top_eigenvector[0], lam)
                loss_list.append(criterion(model_perburbed(inputs), targets).item())

            eigenvalues[f"{labels[i]}"][f"batch_{k}"] = top_eigenvalues
            traces[f"{labels[i]}"][f"batch_{k}"] = np.mean(trace)
            loss_landscape[f"{labels[i]}"][f"batch_{k}"] = loss_list

    return loss_landscape, eigenvalues, traces, eigenvalue_density

def dump_as_json(path, file_name, dict):
    with open(f"{path}/{file_name}.json", "w") as outfile:
        outfile.write(json.dumps(dict, indent=4))

# DO NOT CHANGE (for consistent plotting)
labels = ["Parent 1", "Parent 2", "Fusion (init)", "Fusion (retrain)"]
colors = ["tab:blue", "tab:orange", "tab:red", "tab:purple"]

def plot_loss_landscape(config, loss_landscape, path):
    # Loss Landscape Perturbed by Top EV of Hessian
    figsize = (4.5, 3.5)
    plt.figure(figsize=figsize, dpi=300)

    loss = []
    for key, value in loss_landscape.items():
        if "lambdas" not in key:
            loss.append(list(value.values()))

    loss_np = np.array(loss)
    mean_loss = loss_np.mean(axis=1)
    std_loss = loss_np.std(axis=1)

    # or use loss_landscape['lambdas']
    for i in range(mean_loss.shape[0]):
        plt.fill_between(config['lambdas'], mean_loss[i] + std_loss[i], mean_loss[i] - std_loss[i], alpha = 0.25, color = colors[i])
        plt.plot(config['lambdas'], mean_loss[i], label=labels[i], color = colors[i])

    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Weight Perturbation')
    plt.locator_params(axis='y', nbins=5)
    # plt.suptitle('Cross-Entropy Loss Landscape\n Model parameters perturbed by top Hessian eigenvector')
    plt.savefig(fname=f"{path}/hessian_ev_perturb.pdf", bbox_inches='tight')

def plot_ev_density(eigenvalue_density, path):
    # Density of EVs for each model and batch
    figsize = (4.5, 2.0)
    plt.figure(figsize=figsize, dpi=300)

    density_eigens_total = []
    for i, (model_k, model_v) in enumerate(eigenvalue_density.items()):
        density_eigens = []
        density_weights = []
        for j, (batch_k, [density_eigen, density_weight]) in enumerate(model_v.items()):
            density_eigens += density_eigen
            density_weights += density_weight

        density_eigens_total.append(density_eigens)
        density, grids = density_generate(density_eigens, density_weights)

        plt.plot(grids, density + 1.0e-7, color = colors[i], label= labels[i], alpha = 0.75)
        plt.yscale('log')

        mask = density > 0
        for m in reversed(range(len(mask))):
            if not mask[m]: continue
            else: break
        plt.axvline(grids[m], ymin=0, ymax=1, color = colors[i], linestyle = '-.')
    plt.xlabel('Eigenvalue')
    plt.ylabel('Density [%]')
    plt.yticks([0.000001, 0.001, 10.0])
    plt.axis([np.min(density_eigens_total) - 1, np.max(density_eigens_total) + 1, None, None])
    plt.legend()
    plt.savefig(fname=f"{path}/ev_density_plot.pdf", bbox_inches='tight')

    # # Traces of models
    # traces_np = np.array(traces)
    # mean_traces = traces_np.mean(axis=1)
    # std_traces = traces_np.std(axis=1)

    # # Top EigenValues of models
    # evs_np = np.array(evs)
    # mean_evs = evs_np.mean(axis=1)
    # std_evs = evs_np.std(axis=1)


    # # Store as .csv
    # trace_df = pd.DataFrame({"Model" : ["Parent 1", "Parent 2", "Fusion"],
    #                         "Trace mean": mean_traces,
    #                          "Trace std": std_traces})

    # ev_df = pd.DataFrame({"Parent 1": mean_evs[0],
    #                         "Parent 2" : mean_evs[1],
    #                         "Fusion": mean_evs[2]})

    # trace_df.to_csv(f"{path}/trace.csv", index=False)
    # ev_df.to_csv(f"{path}/ev.csv", index=False)
