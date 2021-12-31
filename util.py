import torch
import numpy as np

from params import *

def convert_model_to_param_list(model):
    '''
    num_of_param=0
    for param in model.state_dict().values():
        num_of_param += torch.numel(param)
    

    params=torch.ones([num_of_param])
    '''
    if torch.typename(model)!='OrderedDict':
        model = model.state_dict()

    idx=0
    params_to_copy_group=[]
    for name, param in model.items():
        num_params_to_copy = torch.numel(param)
        params_to_copy_group.append(param.reshape([num_params_to_copy]).clone().detach())
        idx+=num_params_to_copy

    params=torch.ones([idx])
    idx=0
    for param in params_to_copy_group:    
        for par in param:
            params[idx].copy_(par)
            idx += 1

    return params

def cos_calc_btn_grads(l1, l2):
    return torch.dot(l1, l2)/(torch.linalg.norm(l1)+1e-9)/(torch.linalg.norm(l2)+1e-9)


def cos_calc(n1, n2):
    l1 = convert_model_to_param_list(n1)
    l2 = convert_model_to_param_list(n2)
    return cos_calc_btn_grads(l1, l2)

def inv_grad_test(model):
    for name, param in model.named_parameters():
        print(name, torch.isfinite(param.grad).all())

def calcDiff(network, network2):
    return sum((x - y).abs().sum() for x, y in zip(network.state_dict().values(), network2.state_dict().values()))

