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
	fl = CustomFL(
		n_iter=n_iter, n_epochs=n_epochs, poison_starts_at_iter=poison_starts_at_iter, 
		validation_starts_at_iter=validation_starts_at_iter, 
		num_of_benign_nets=num_of_workers-num_of_mal_workers, num_of_mal_nets=num_of_mal_workers,      inertia=inertia, momentum=momentum, 
		attack_type=attack_type, scale_up=scale_up, minimizeDist=minimizeDist
	)

	group_report = [{} for _ in range(10)]
	if flt_aggr:
		group_report.append({})
	for id in range(len(train_loaders[0])):
		(_, _, train_loader) = train_loaders[0][id]
		lsr = get_label_skew_ratios(train_loader.dataset)
		fl.debug_log['label_skew_ratio'].append((id, lsr))
		group_report[copylist[id]][id] = lsr[copylist[id]%10]

	fl.debug_log['label_skew_group_report'] = group_report
	for i in range(10):
		print(f'Group {i}')
		print(group_report[i])

	# for i in range(len(fl.debug_log['label_skew_ratio'])):
	# 	(_, lsr) = fl.debug_log['label_skew_ratio'][i]
	# 	print(i, lsr)

	fl.train()

	with open(f'{output_filename}_{begin_time}.txt'.replace(':', '-'), 'wb') as f:
		pickle.dump(fl.debug_log, f)

	f.close()