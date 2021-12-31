import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

import pickle

from CustomFL import CustomFL

from params import *

if __name__ == '__main__':
	fl = CustomFL(n_iter=n_iter, n_epochs=n_epochs, poison_starts_at_iter=poison_starts_at_iter, num_of_benign_nets=num_of_workers-num_of_mal_workers, num_of_mal_nets=num_of_mal_workers, 
	              inertia=inertia, momentum=momentum,
	              attack_type=attack_type, scale_up=scale_up, minimizeDist=minimizeDist
	)

	for id in range(len(train_loaders[0])):
		(_, _, train_loader) = train_loaders[0][id]
		fl.debug_log['label_skew_ratio'].append((id, get_label_skew_ratios(train_loader.dataset)))
	
	fl.train()

	with open(f'output_file_dec30_mal_{num_of_mal_workers}_aa0_{aa0}_all_{num_of_workers}_iters_{n_iter}_epochs_{n_epochs}.txt', 'wb') as f:
		pickle.dump(fl.debug_log, f)

	f.close()