import torch.nn.functional as F

def new_loss(output, target):
    loss = F.nll_loss(output, target)

    return loss