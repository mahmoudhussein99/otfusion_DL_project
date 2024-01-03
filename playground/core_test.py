import numpy as np
import pandas as pd

import torch 
from torchvision import datasets, transforms

from pyhessian import hessian # Hessian computation

from utils_pyh import * # get the dataset

import matplotlib.pyplot as plt

import sys

sys.path.append("../")
import cifar.models

import copy
import time

USE_CUDA = False
BATCH_SIZE = 8
NUM_BATCHES = 2
TOP_EV = 3
TIME = time.strftime("_%m-%d_%H-%M-%S")


def get_params(model_orig,  model_perb, direction, alpha):
    for m_orig, m_perb, d in zip(model_orig.parameters(), model_perb.parameters(), direction):
        # print("param: ", m_orig.data.shape)
        # print("direction: ", d.shape)
        m_perb.data = m_orig.data + alpha * d
    return model_perb

def evaluate_hessian(models, loader):
    # lambda is a small scalar that we use to perturb the model parameters along the eigenvectors 
    lams = np.linspace(-0.5, 0.5, 21).astype(np.float32)
    labels = ["parent1", "parent2", "fusion"]
    
    criterion = torch.nn.CrossEntropyLoss()

    ev_total = []
    trace_total = []
    loss_list_total = []
    for k, (inputs, targets) in enumerate(loader):
        if k >= NUM_BATCHES: break
        
        ev_models = []
        trace_models = []
        loss_list_models = []
        
        for i, model in enumerate(models):
            hessian_comp = hessian(model, criterion, data=(inputs, targets), cuda=USE_CUDA) # cuda=True
            trace = hessian_comp.trace()

            top_eigenvalues, top_eigenvector = hessian_comp.eigenvalues(top_n = TOP_EV)
            loss_list = []
            
            # create a copy of the model
            model_perb = copy.deepcopy(model)
            model_perb.eval()
            
            if USE_CUDA:
                model_perb = model_perb.cuda()

            for lam in lams:
                model_perb = get_params(model, model_perb, top_eigenvector[0], lam) # Largest eigenvalue!
                loss_list.append(criterion(model_perb(inputs), targets).item())
            
            ev_models.append(top_eigenvalues)
            loss_list_models.append(loss_list)
            trace_models.append(np.mean(trace))
        
        ev_total.append(ev_models)
        loss_list_total.append(loss_list_models)
        trace_total.append(trace_models)

    return loss_list_total, ev_total, trace_total

def output_results(result, evs, traces):
    # Traces of models
    traces_np = np.array(traces)
    mean_traces = traces_np.mean(axis=0)
    std_traces = traces_np.std(axis=0)

    # Top 3 Eigen Values models
    evs_np = np.array(evs)
    mean_evs = evs_np.mean(axis=0)
    std_evs = evs_np.std(axis=0)

    trace_df = pd.DataFrame({"Trace mean": mean_traces, 
                             "Trace std": std_traces})
    

    ev_df = pd.DataFrame({"Parent 1": mean_evs[0], 
                             "Parent 2": mean_evs[1], 
                             "Fusion": mean_evs[2], 
                             })

    trace_df.to_csv(f"./trace{TIME}.csv")
    ev_df.to_csv(f"./ev{TIME}.csv")

    # Loss Landscape Perturbed by Top EV of Hessian
    result_np = np.array(result)
    mean = result_np.mean(axis=0)
    std = result_np.std(axis=0)

    labels = ["parent1", "parent2", "fusion"]
    colors = ["blue", "orange", "red"]
    
    lams = np.linspace(-0.5, 0.5, 21).astype(np.float32)

    for i in range(mean.shape[0]):
        plt.fill_between(lams, mean[i] + std[i], mean[i] - std[i], alpha = 0.25, color = colors[i])
        plt.plot(lams, mean[i], label=labels[i], color = colors[i])

    plt.legend()
    plt.ylabel('Loss')
    plt.xlabel('Perturbation')
    plt.title('Loss landscape perturbed based on top Hessian eigenvector')
    plt.savefig(fname=f"./perturbation{TIME}.png")

def main():
    device = torch.device('cuda') if USE_CUDA  else torch.device('cpu')

    # Load Models
    parent1 = cifar.models.VGG('VGG11', 10, batch_norm=False, bias=False, relu_inplace=True)
    parent2 = cifar.models.VGG('VGG11', 10, batch_norm=False, bias=False, relu_inplace=True)
    fusion = cifar.models.VGG('VGG11', 10, batch_norm=False, bias=False, relu_inplace=True)

    state1 = torch.load("../cifar_models/model_0/best.checkpoint", map_location=device)
    state2 = torch.load("../cifar_models/model_1/best.checkpoint", map_location=device)
    state3 = torch.load("../cifar_models/fusion_retraining/best.checkpoint", map_location=device)

    parent1.load_state_dict(state1['model_state_dict'])
    parent2.load_state_dict(state2['model_state_dict'])
    fusion.load_state_dict(state3['model_state_dict'])

    #Dataset 
    test_loader = torch.utils.data.DataLoader(
                    datasets.CIFAR10('../data/', train=False, download=True,
                                            transform=transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(
                                                    # (mean_ch1, mean_ch2, mean_ch3), (std1, std2, std3)
                                                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                            ])),
                    batch_size = BATCH_SIZE, shuffle=True
                )
    
    result, evs, traces = evaluate_hessian([parent1, parent2, fusion], test_loader)
    
    output_results(result, evs, traces)

    return

if __name__ == "__main__":
    main()