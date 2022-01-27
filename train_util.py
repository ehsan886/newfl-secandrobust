import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

from tqdm import tqdm

from params import *
from losses import *
from poison_util import *

loss_func=nn.CrossEntropyLoss()

def train(network, optimizer, epoch):
	network.train()

	if network.network_id!=-1:
	  (_, temp_train_loader)=train_loaders[network.network_id]
	else:
	  temp_train_loader=train_loader

	for batch_idx, (data, target) in enumerate(temp_train_loader):
		if network.network_id>=1:
		    data, target, poison_num = get_poison_batch((data, target))
	optimizer.zero_grad()
	output = network(data)
	loss_func=nn.CrossEntropyLoss()
	loss = loss_func(output, target)
	loss.backward()
	#inv_grad_test(network)
	optimizer.step()
	if batch_idx % log_interval == 0:
		print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
		epoch, batch_idx * len(data), len(train_loader.dataset),
		100. * batch_idx / len(train_loader), loss.item()))
		train_losses.append(loss.item())
		train_counter.append(
		(batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
		torch.save(network.state_dict(), 'model.pth')
		torch.save(optimizer.state_dict(), 'optimizer.pth')
	if network.network_id>=1:
 	 	sys.exit()


def train_net(network, optimizer, trainloader, epoch, poisonNow=False, print_flag=False, tqdm_disable=True, attack_type='backdoor'):
	for batch_idx, (data, target) in enumerate(tqdm(trainloader, disable=tqdm_disable)):
		if poisonNow:
			data, target, poison_num = get_poison_batch_special_label_flip((data, target))
		else:
			data, target = get_batch((data, target))
		optimizer.zero_grad()
		output = network(data)
		loss = loss_func(output, target)
		loss.backward()
		#inv_grad_test(network)
		optimizer.step()
		if batch_idx % log_interval == 0:
			if print_flag:
				print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(trainloader.dataset),
				100. * batch_idx / len(trainloader), loss.item()))
			train_losses.append(loss.item())
			train_counter.append(
				(batch_idx*64) + ((epoch-1)*len(trainloader.dataset)))


def validation_test(network, test_loader, is_poisonous=False, tqdm_disable=True):
	network.eval()
	correct = 0
	with torch.no_grad():
		for data, target in tqdm(test_loader, disable=tqdm_disable):
			if is_poisonous:
				data, target, poison_num = get_poison_batch_special_label_flip((data, target))
			else:
				data, target = get_batch((data, target))
			output = network(data)
			loss_func=nn.CrossEntropyLoss()
			pred = output.data.max(1, keepdim=True)[1]
			correct += pred.eq(target.data.view_as(pred)).sum()
	
	return 100. * correct / len(test_loader.dataset)

def test(network):
	network.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in tqdm(test_loader):
			data, target = get_batch((data, target))
			output = network(data)
			loss_func=nn.CrossEntropyLoss()
			test_loss += loss_func(output, target).item()
			pred = output.data.max(1, keepdim=True)[1]
			correct += pred.eq(target.data.view_as(pred)).sum()
	test_loss /= len(test_loader.dataset)
	test_losses.append(test_loss)
	print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
	test_loss, correct, len(test_loader.dataset),
	100. * correct / len(test_loader.dataset)))
	return 100. * correct / len(test_loader.dataset)





def test_label_flip(network, print_flag=False, tqdm_disable=True):
	network.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in tqdm(target_class_test_loader, disable=tqdm_disable):
		    data, target = get_batch((data, target))
		    output = network(data)
		    loss_func=nn.CrossEntropyLoss()
		    test_loss += loss_func(output, target).item()
		    pred = output.data.max(1, keepdim=True)[1]
		    correct += pred.eq(target.data.view_as(pred)).sum()
	test_loss /= len(target_class_test_loader.dataset)
	test_losses.append(test_loss)
	if print_flag:
		print('\nTarget Class Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(target_class_test_loader.dataset),
		100. * correct / len(target_class_test_loader.dataset)))
	return 100. * correct / len(target_class_test_loader.dataset)


def backdoor_test(network, tqdm_disable=True):
	network.eval()
	test_loss = 0
	correct = 0
	with torch.no_grad():
		for data, target in tqdm(test_loader, disable=tqdm_disable):
			data, target, poison_num = get_poison_batch((data, target), evaluation=True, attack_type='backdoor')
			output = network(data)
			loss_func=nn.CrossEntropyLoss()
			test_loss += loss_func(output, target).item()
			pred = output.data.max(1, keepdim=True)[1]
			correct += pred.eq(target.data.view_as(pred)).sum()
	test_loss /= len(test_loader.dataset)
	test_losses.append(test_loss)
	print('\nBackdoor Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
	test_loss, correct, len(test_loader.dataset),
	100. * correct / len(test_loader.dataset)))