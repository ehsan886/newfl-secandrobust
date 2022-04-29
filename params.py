import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

from collections import defaultdict

import argparse
import datetime

import scenario_dict

parser = argparse.ArgumentParser()
parser.add_argument('--attacker_at_0', dest='aa0', default=0)
parser.add_argument('--server_pct', dest='server_pct', default=0.1)
parser.add_argument('--max_exec_min', dest='max_exec_min', default=5)
parser.add_argument('--output_filename', dest='output_filename', default='temp_exp')
parser.add_argument('--num_of_mal_workers', dest='num_of_mal_workers', default=20)
parser.add_argument('--n_iter', dest='n_iter', default=30)
parser.add_argument('--poison_starts_at_iter', dest='poison_starts_at_iter', default=0)
parser.add_argument('--bias', dest='bias', default=0.5)
parser.add_argument('--upper_bound_offset', dest='upper_bound_offset', default=0)
parser.add_argument('--clustering_on', dest='clustering_on', default=1)
parser.add_argument('--server_priv_att_iter', dest='server_priv_att_iter', default=-1)
parser.add_argument('--validation_starts_at_iter', dest='validation_starts_at_iter', default=10)
parser.add_argument('--dataset', dest='dataset_name', default='fmnist')
parser.add_argument('--iterative_k', dest='iterative_k', default=10)
parser.add_argument('--save_local_models_opt', dest='save_local_models_opt', default=0)
parser.add_argument('--evasion_type', dest='evasion_type', default=None)

parser.add_argument('--target_class', dest='target_class', default=0)
parser.add_argument('--scenario_dict_on', dest='scenario_dict_on', default=0)

args = parser.parse_args()

aa0 = int(args.aa0)
server_pct = float(args.server_pct)
max_exec_min = datetime.timedelta(minutes= float(args.max_exec_min))
output_filename = args.output_filename
dataset_name = args.dataset_name
evasion_type = args.evasion_type
iterative_k = int(args.iterative_k)
save_global_model = False
save_local_models_opt = int(args.save_local_models_opt)
target_class = int(args.target_class)
num_of_mal_workers=int(args.num_of_mal_workers)
nuiid = True
scenario_dict_on = int(args.scenario_dict_on)

if scenario_dict_on:
    configs = scenario_dict[output_filename]
    num_of_mal_workers = configs[0]
    aa0 = configs[1]
    target_class = configs[2]
    dataset_name = configs[3]
    if configs[4] == 'uniform':
        nuiid = False
    evasion_type = configs[5]


begin_time = datetime.datetime.now()

poison_dict = dict()
poison_dict['poison_delta'] = 0.1
poison_dict['poison_pattern'] = [[23,25], [24,24],[25,23],[25,25]]
poison_dict['poisoning_per_batch'] = 80
poison_dict['poison_label_swap'] = 0

#@title
import torchvision
import random

batch_size_train = 100
batch_size_test = 1000
learning_rate = 0.01
log_interval = 10

### important hyperparameters
flt_aggr=True
clustering_on = int(args.clustering_on)
if clustering_on:
    flt_aggr=False
num_of_workers=100
if flt_aggr:
    num_of_workers+=1
n_iter=int(args.n_iter)
# n_iter = 250
n_epochs=1
poison_starts_at_iter=int(args.poison_starts_at_iter)
# poison_starts_at_iter=12
validation_starts_at_iter=int(args.validation_starts_at_iter)
inertia=0.1
momentum=0.1
attack_type='label_flip'
scale_up=False
minimizeDist=False

# target_class=0

iid = False
bias = float(args.bias)
num_of_distributions = int(num_of_workers/10)+1
server_priv_att_iter = int(args.server_priv_att_iter)
# num_of_workers_in_distribs = num_of_workers * np.random.dirichlet(np.array(num_of_distributions * [3.0]))
# num_of_workers_in_distribs = [int(val) for val in num_of_workers_in_distribs]
# while 0 in num_of_workers_in_distribs:
#     num_of_workers_in_distribs.remove(0)
# num_of_workers_in_distribs.append(num_of_workers-sum(num_of_workers_in_distribs))
# print(num_of_workers_in_distribs, sum(num_of_workers_in_distribs))
# num_of_distributions = len(num_of_workers_in_distribs)
# copylist = []
# for i in range(len(num_of_workers_in_distribs)):
#     copylist += num_of_workers_in_distribs[i]*[i]
# random.shuffle(copylist)
# print(copylist)

label_skew_ratios=[]

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

random_seed = 1
torch.backends.cudnn.enabled = False
torch.manual_seed(random_seed)

train_losses = []
train_counter = []
test_losses = []
#test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

#@title
from torchvision import datasets, transforms

dataPath = ''

import random

train_loaders=[]

transform = transforms.Compose([transforms.ToTensor(),
    ### if dataset is mnist
    transforms.Normalize((0.1307,), (0.3081,))])
    ### if dataset is cifar
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

if dataset_name=='mnist':
    train_dataset = datasets.MNIST('', train=True, download=True,
                                transform=transform)
    test_dataset = datasets.MNIST('', train=False, transform=transform)
else:
    train_dataset = datasets.FashionMNIST('', train=True, download=True,
                                transform=transform)
    test_dataset = datasets.FashionMNIST('', train=False, transform=transform)

train_dataset.targets[train_dataset.targets==0] = 10
train_dataset.targets[train_dataset.targets==target_class] = 0
train_dataset.targets[train_dataset.targets==10] = target_class

test_dataset.targets[test_dataset.targets==0] = 10
test_dataset.targets[test_dataset.targets==target_class] = 0
test_dataset.targets[test_dataset.targets==10] = target_class

target_class = 0

test_loader = torch.utils.data.DataLoader(
  test_dataset,
  batch_size=batch_size_test, shuffle=True)

all_range = list(range(len(train_dataset)))
random.shuffle(all_range)

def get_train_iid(all_range, model_no, iter_no):
    """
    This method equally splits the dataset.
    :param params:
    :param all_range:
    :param model_no:
    :return:
    """

    data_len_for_iter = int(len(train_dataset) / n_iter)
    data_len = int(data_len_for_iter/num_of_workers)
    sub_indices_for_iter = all_range[iter_no * data_len_for_iter: (iter_no + 1) * data_len_for_iter]
    sub_indices = sub_indices_for_iter[model_no * data_len: (model_no + 1) * data_len ]
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                        batch_size=batch_size_train,
                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(sub_indices)
                                        )
    return train_loader

def get_train_noniid(indices):
    """
    This method is used along with Dirichlet distribution
    :param params:
    :param indices:
    :return:
    """
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                        batch_size=int(len(train_dataset)/num_of_workers),
                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                            indices))
    return train_loader

def poison_test_dataset(test_dataset, batch_size):
    # logger.info('get poison test loader')
    # delete the test data with target label
    test_classes = {}
    for ind, x in enumerate(test_dataset):
        _, label = x
        if label in test_classes:
            test_classes[label].append(ind)
        else:
            test_classes[label] = [ind]

    range_no_id = list(range(0, len(test_dataset)))
    for image_ind in test_classes[poison_dict['poison_label_swap']]:
        if image_ind in range_no_id:
            range_no_id.remove(image_ind)
    poison_label_inds = test_classes[poison_dict['poison_label_swap']]

    return torch.utils.data.DataLoader(test_dataset,
                        batch_size=batch_size,
                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                            range_no_id)), \
            torch.utils.data.DataLoader(test_dataset,
                                        batch_size=batch_size,
                                        sampler=torch.utils.data.sampler.SubsetRandomSampler(
                                            poison_label_inds))
    
def sample_dirichlet_train_data(no_participants=num_of_workers, dataset=train_dataset, alpha=0.9, copylist=np.arange(num_of_workers)):
    """
        Input: Number of participants and alpha (param for distribution)
        Output: A list of indices denoting data in CIFAR training set.
        Requires: dataset_classes, a preprocessed class-indice dictionary.
        Sample Method: take a uniformly sampled 10-dimension vector as parameters for
        dirichlet distribution to sample number of images in each class.
    """

    dataset_classes = {}
    for ind, x in enumerate(dataset):
        _, label = x
        #if ind in self.params['poison_images'] or ind in self.params['poison_images_test']:
        #    continue
        if label in dataset_classes:
            dataset_classes[label].append(ind)
        else:
            dataset_classes[label] = [ind]
    class_size = len(dataset_classes[0])
    per_participant_list = defaultdict(list)
    no_classes = len(dataset_classes.keys())

    for n in range(no_classes):
        random.shuffle(dataset_classes[n])
        num_of_non_iid_participants = len(np.unique(copylist))
        sampled_probabilities = np.random.dirichlet(
            np.array(num_of_non_iid_participants * [alpha]))
        new_list = []
        for ip in copylist:
            #new_list.append(np.random.normal(loc=sampled_probabilities[ip], scale=0.005))
            new_list.append(sampled_probabilities[ip])
        sampled_probabilities = class_size * np.array(new_list)/np.sum(np.array(new_list))
        sigmas = 0.0 * sampled_probabilities
        sampled_probabilities = np.random.normal(sampled_probabilities, scale=sigmas)
        #print(sampled_probabilities)
        label_skew_ratios.append(sampled_probabilities)
        for user in range(no_participants):
            no_imgs = int(round(sampled_probabilities[user]))
            sampled_list = dataset_classes[n][:min(len(dataset_classes[n]), no_imgs)]
            per_participant_list[user].extend(sampled_list)
            dataset_classes[n] = dataset_classes[n][min(len(dataset_classes[n]), no_imgs):]

    return per_participant_list

def get_label_skew_ratios(dataset, num_of_classes=10):
    dataset_classes = {}
    for ind, x in enumerate(dataset):
        _, label = x
        #if ind in self.params['poison_images'] or ind in self.params['poison_images_test']:
        #    continue
        if label in dataset_classes:
            dataset_classes[label] += 1
        else:
            dataset_classes[label] = 1
    for key in dataset_classes.keys():
    	# dataset_classes[key] = dataset_classes[key] 

        dataset_classes[key] = float("{:.2f}".format(dataset_classes[key]/len(dataset)))
    return dataset_classes

def assign_data(train_data, bias, ctx, num_labels=10, num_workers=100, server_pc=100, p=0.01, server_case2_cls=0, dataset="FashionMNIST", seed=1, flt_aggr=True):
    # assign data to the clients
    other_group_size = (1 - bias) / (num_labels - 1)
    worker_per_group = num_workers / num_labels

    #assign training data to each worker
    each_worker_data = [[] for _ in range(num_workers)]
    each_worker_label = [[] for _ in range(num_workers)]   
    server_data = []
    server_label = []
    
    # compute the labels needed for each class
    real_dis = [1. / num_labels for _ in range(num_labels)]
    samp_dis = [0 for _ in range(num_labels)]
    num1 = int(server_pc * p)
    samp_dis[server_case2_cls] = num1
    average_num = (server_pc - num1) / (num_labels - 1)
    resid = average_num - np.floor(average_num)
    sum_res = 0.
    for other_num in range(num_labels - 1):
        if other_num == server_case2_cls:
            continue
        samp_dis[other_num] = int(average_num)
        sum_res += resid
        if sum_res >= 1.0:
            samp_dis[other_num] += 1
            sum_res -= 1
    samp_dis[num_labels - 1] = server_pc - np.sum(samp_dis[:num_labels - 1])

    print('samp_dis', samp_dis)

    # privacy experiment only
    server_additional_label_0_samples_counter = 0    
    server_add_data=[]
    server_add_label=[]

    # randomly assign the data points based on the labels
    server_counter = [0 for _ in range(num_labels)]
    for _, (x, y) in enumerate(train_data):
        '''
        if dataset == "FashionMNIST":
            x = x.as_in_context(ctx).reshape(1,1,28,28)
        else:
            raise NotImplementedError
        y = y.as_in_context(ctx)
        '''

        upper_bound = y * (1. - bias) / (num_labels - 1) + bias
        lower_bound = y * (1. - bias) / (num_labels - 1)

        # print(y, upper_bound, lower_bound)

        # experiment 2 only
        # upper_bound_offset = 0.4 if y==0 else 0
        upper_bound_offset = float(args.upper_bound_offset) if y==0 else 0

        rd = np.random.random_sample()


        other_group_size = (1 - upper_bound - upper_bound_offset + lower_bound) / (num_labels - 1)

        if rd > upper_bound + upper_bound_offset:
            worker_group = int(np.floor((rd - upper_bound - upper_bound_offset) / other_group_size) + y + 1)
        elif rd < lower_bound:
            worker_group = int(np.floor(rd / other_group_size))
        # experiment 2 only
        elif rd > upper_bound:
            continue
        else:
            worker_group = y
        # if y==0 and y != worker_group:
        #     print(rd, y, worker_group)

        if server_additional_label_0_samples_counter < 100 and y==0 and server_priv_att_iter!=-1:
            server_add_data.append(x)
            server_add_label.append(y)
            server_additional_label_0_samples_counter += 1
        elif server_counter[int(y)] < samp_dis[int(y)] and flt_aggr:
            server_data.append(x)
            server_label.append(y)
            server_counter[int(y)] += 1
        else:
            rd = np.random.random_sample()
            selected_worker = int(worker_group * worker_per_group + int(np.floor(rd * worker_per_group)))
            each_worker_data[selected_worker].append(x)
            each_worker_label[selected_worker].append(y)
   
    #     server_data = nd.concat(*server_data, dim=0)
    #     server_label = nd.concat(*server_label, dim=0)
    
    
#     each_worker_data = [nd.concat(*each_worker, dim=0) for each_worker in each_worker_data] 
#     each_worker_label = [nd.concat(*each_worker, dim=0) for each_worker in each_worker_label]
    

#     # randomly permute the workers
#     random_order = np.random.RandomState(seed=seed).permutation(num_workers)
#     each_worker_data = [each_worker_data[i] for i in random_order]
#     each_worker_label = [each_worker_label[i] for i in random_order]
    

    return server_data, server_label, each_worker_data, each_worker_label, server_add_data, server_add_label


# if iid:
#     train_loaders=[]
#     for i in range(n_iter):
#         train_loaders.append([(i, pos, get_train_iid(all_range, pos, i))
#                                 for pos in range(num_of_workers)])
# else:
#     indices_per_participant = sample_dirichlet_train_data(
#         num_of_workers,
#         #dataset= torch.utils.data.Subset(train_dataset, list(range(240))),
#         alpha=0.95,
#         copylist=copylist)
#     train_loaders = [(-1, pos, get_train_noniid(indices)) for pos, indices in
#                     indices_per_participant.items()]
#     train_loaders = n_iter * [train_loaders]

def assign_data_nonuniform(train_data, bias, ctx, num_labels=10, num_workers=100, server_pc=100, p=0.01, server_case2_cls=0, dataset="FashionMNIST", seed=1, flt_aggr=True):
    server_data, server_label, each_worker_data, each_worker_label, server_add_data, server_add_label = assign_data(train_data, bias, ctx, num_labels, 10, server_pc, p, server_case2_cls, dataset, seed, flt_aggr)
    ewd = [[] for _ in range(num_workers)]
    ewl = [[] for _ in range(num_workers)]
    # group_sizes = [np.random.randint(5, 16) for i in range(9)]
    # group_sizes.append(num_workers-sum(group_sizes))
    # if group_sizes[-1] < 5 or group_sizes[-1] > 15:
    #     avg_last_2 = (group_sizes[-1] + group_sizes[-2])/2
        
    copylist = []
    for i, group_size in enumerate(group_sizes):
        for _ in range(group_size):
            copylist.append(i)

    for i in range(len(each_worker_data)):
        group_size = group_sizes[i]
        group_frac = 1./group_size
        label_data = np.array(each_worker_label[i])
        i_indices = np.where(label_data==i)[0].tolist()
        not_i_indices = np.where(label_data!=i)[0].tolist()
        split_map_for_i = []
        split_map_for_i.append(0)
        split_map_for_not_i = [0]
        for ii in range(1, group_size):
            split_ratio_for_i = np.random.normal(ii*group_frac, group_frac/10)
            split_ratio_for_not_i = ii*group_frac*2 - split_ratio_for_i
            split_map_for_i.append(int(split_ratio_for_i*len(i_indices)))
            split_map_for_not_i.append(int(split_ratio_for_not_i*len(not_i_indices)))
        split_map_for_i.append(len(i_indices))
        split_map_for_not_i.append(len(not_i_indices))
        i_indices_list = [i_indices[split_map_for_i[ii]:split_map_for_i[ii+1]] for ii in range(group_size)]
        not_i_indices_list = [not_i_indices[split_map_for_not_i[ii]:split_map_for_not_i[ii+1]] for ii in range(group_size)]
        indice_map = [0]*len(each_worker_data[i])
        for ii in range(group_size):
            for iii in i_indices_list[ii]:
                indice_map[iii] = ii 
            for iii in not_i_indices_list[ii]:
                indice_map[iii] = ii 
        size_of_group = int(len(each_worker_data[i])/10)
        stop_val = 10 * size_of_group
        for idx in range(len(each_worker_data[i])):
            ewd[sum(group_sizes[:i]) + indice_map[idx]].append(each_worker_data[i][idx])
            ewl[sum(group_sizes[:i]) + indice_map[idx]].append(each_worker_label[i][idx])
    return server_data, server_label, ewd, ewl, server_add_data, server_add_label


if flt_aggr:
    copylist=[int(np.floor(i/((num_of_workers-1)/10))) for i in range(num_of_workers-1)]
    copylist.append(copylist[-1]+1)
else:
    copylist=[int(np.floor(i/((num_of_workers)/10))) for i in range(num_of_workers)]

# mal_indices=[19, 28, 37, 46, 55, 64, 73, 82, 91]
# mal_indices=[18, 19, 27, 28, 36, 37, 45, 46, 54, 55, 63, 64, 72, 73, 81, 82, 90, 91]

if nuiid:
    group_sizes = [10, 10, 5, 15, 5, 15, 5, 15, 5, 15]
else:
    group_sizes = [10 for _ in range(10)]
copylist = []
for i, group_size in enumerate(group_sizes):
    for _ in range(group_size):
        copylist.append(i)
if flt_aggr:
    copylist.append(copylist[-1]+1)

sd, sl, ewd, ewl, sad, sal = assign_data_nonuniform(train_dataset, bias, None, p=server_pct, flt_aggr=flt_aggr)

if flt_aggr:
    ewd.append(sd)
    ewl.append(sl)

group_0_list=np.arange(target_class*10, (target_class+1)*10)
other_group_list=np.union1d(np.arange(0, group_0_list[0]), np.arange(group_0_list[-1]+1, 100))
np.random.shuffle(group_0_list)
print(other_group_list)
np.random.shuffle(other_group_list)
# print(group_0_list[:aa0])
mal_indices = np.sort(np.array(group_0_list[:aa0].tolist() + other_group_list[:num_of_mal_workers-aa0].tolist()))
print(mal_indices)

for index in mal_indices:
	ew_d = ewd[index]
	ew_l = ewl[index]
	ew_c = copylist[index]
	ewd.append(ew_d)
	ewl.append(ew_l)
	copylist.append(ew_c)

for id in range(len(mal_indices)):
	index = mal_indices[len(mal_indices)-id-1]
	del ewd[index]
	del ewl[index]
	del copylist[index]

print('copylist ', copylist)

if flt_aggr and server_priv_att_iter!=-1:
    ewd.append(sad)
    ewl.append(sal)

# from scipy import stats

# for id, mal in enumerate(range(num_of_workers-num_of_mal_workers, num_of_workers)):
# 	mals_benign_brothers = np.where(np.array(copylist)==copylist[mal])
# 	mals_benign_brothers_clusters = [copylist[(iid+1)%len(copylist)] for iid in mals_benign_brothers[0]]
# 	print(mals_benign_brothers, mals_benign_brothers_clusters)
# 	print(stats.mode(mals_benign_brothers_clusters)[0])

first_target_group_mal_index = np.where(np.array(copylist)==target_class)[0][aa0]
print(first_target_group_mal_index)

train_loaders = []
for id_worker in range(len(ewd)):
    dataset_per_worker=[]
    for idx in range(len(ewd[id_worker])):
        dataset_per_worker.append((ewd[id_worker][idx], ewl[id_worker][idx]))
    if len(dataset_per_worker) != 0:
        train_loader = torch.utils.data.DataLoader(dataset_per_worker, batch_size=batch_size_train, shuffle=True)
        train_loaders.append((-1, id_worker, train_loader))
        
#train_loaders = [(-1, idx, torch.utils.data.DataLoader(ew, batch_size=batch_size_train, shuffle=True)) for idx, ew in enumerate(ewd)]
train_loaders = n_iter * [train_loaders]


#privacy experiment only
# for iter in range(1, n_iter):
#     if iter%10 == 0:
#         train_loaders[iter][num_of_workers-num_of_mal_workers-1] = train_loaders[iter][num_of_workers]

target_class_test_data=[]
for _, (x, y) in enumerate(test_dataset):
	if y==target_class:
		target_class_test_data.append((x, y))

print(len(target_class_test_data))

target_class_test_loader = torch.utils.data.DataLoader(target_class_test_data, batch_size=batch_size_test, shuffle=True)