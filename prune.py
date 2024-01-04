import torch
import torch.nn.utils.prune

def prune_model(model,fraction=0.5,prune_type='unstructured'):
    for name, module in model.named_modules():
        if isinstance(module,torch.nn.Conv2d):
            if prune_type == 'unstructured':
                torch.nn.utils.prune.l1_unstructured(module,name='weight',amount=fraction)
            elif prune_type == 'structured':
                torch.nn.utils.prune.ln_structured(module, name='weight', amount=fraction, n=1, dim=3)
            else:
                raise NotImplementedError(f'prune_type {prune_type} not recognized')
            torch.nn.utils.prune.remove(module, 'weight')
    print('---------let\'s see result after pruning-------------' )
    print(dict(model.named_buffers()).keys())



