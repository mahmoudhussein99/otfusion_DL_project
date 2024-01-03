import torch
from torch import nn

def prune_model(model,fraction=0.5):
    for name, module in model.named_modules():
        if isinstance(module,torch.nn.Conv2d):
            nn.utils.prune.l1_unstructured(module,name='weight',amount=fraction)
    print('---------let\'s see result after pruning-------------' )
    print(dict(model.named_buffers()).keys())



