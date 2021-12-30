import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

from CustomFL import CustomFL

from params import *

if __name__ == '__main__':
	fl = CustomFL(n_iter=n_iter, n_epochs=n_epochs, poison_starts_at_iter=poison_starts_at_iter, num_of_benign_nets=num_of_workers-num_of_mal_workers, num_of_mal_nets=num_of_mal_workers, 
	              inertia=inertia, momentum=momentum,
	              attack_type=attack_type, scale_up=scale_up, minimizeDist=minimizeDist
	)
	fl.train()