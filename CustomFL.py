import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

from models import CNN

from params import *
from optim import *
from train_util import *
from util import *


def get_scaled_up_grads(glob_net, networks, self=None, iter=-1):
    nets_grads=[]
    clean_server_grad=None
    grads=[]

    for i in range(len(networks)):
        grad_net=CNN().to(device)
        grad_net.to(device)
        grad_net.copy_params(networks[i].state_dict())
        nets_grads.append(grad_net)
        
        grad_net.calc_grad(glob_net.state_dict())
        grads.append(convert_model_to_param_list(grad_net))

    scaled_grad=CNN().to(device)
    scaled_grad.set_param_to_zero()
    scaled_grad.aggregate([n.state_dict() for n in nets_grads], aggr_weights=[-1]*(len(networks)-1)+[len(networks)])

    self.log.append((iter, 'Cos_sim btn scaled grad and clean server grad', 'get_scaled_up_grads', cos_calc(scaled_grad, nets_grads[-2])))
    print(self.log[-1])
    self.log.append((iter, 'Cos_sim btn mal grad and clean server grad', 'get_scaled_up_grads', cos_calc(nets_grads[-1], nets_grads[-2])))
    print(self.log[-1])

    scaled_grad.aggregate([glob_net.state_dict()], aggr_weights=[1])
    return scaled_grad

    
class CustomFL:
    def __init__(self, num_of_benign_nets=1, num_of_mal_nets=1, inertia=0.1, n_iter=10,
                 n_epochs=3, poison_starts_at_iter=3, learning_rate=0.1, momentum=0, weight_decay=0.1,
                 attack_type='label_flip', scale_up=False, minimizeDist=True):
        self.global_net = CNN().to(device)
        self.global_net_optim = SGD(self.global_net.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        self.global_net.to(device)
        self.benign_nets = []
        self.benign_net_optims = []
        for i in range(num_of_benign_nets):
            network = CNN(net_id=i)
            network.copy_params(self.global_net.state_dict())
            optim = SGD(network.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay, inertia=inertia)
            network.to(device)
            self.benign_nets.append(network)
            self.benign_net_optims.append(optim)
        self.mal_nets = []
        self.mal_net_optims = []
        for i in range(num_of_mal_nets):
            network = CNN(is_malicious=True, net_id=i)
            network.copy_params(self.global_net.state_dict())
            optim = SGD(network.parameters(), lr=learning_rate,
                        momentum=momentum, weight_decay=weight_decay, inertia=inertia, minimizeDist=minimizeDist)
            network.to(device)
            self.mal_nets.append(network)
            self.mal_net_optims.append(optim)

        self.current_iter=0
        self.num_of_benign_nets=num_of_benign_nets
        self.num_of_mal_nets=num_of_mal_nets
        self.inertia_rate=inertia
        self.n_iter=n_iter
        self.n_epochs=n_epochs
        self.learning_rate=learning_rate
        self.momentum=momentum
        self.poison_starts_at_iter=poison_starts_at_iter
        self.weight_decay=weight_decay
        self.attack_type=attack_type
        self.scale_up=scale_up

        self.log=[]
        self.debug_log={}
        self.debug_log['cluster']=[]
        self.debug_log['cluster_without_running_avg']=[]
        self.debug_log['cluster_mal']=[]
        self.debug_log['cluster_mal_wra']=[]
        self.debug_log['cluster_labels']=[]
        self.debug_log['coses']=[]
        self.debug_log['target_label_acc']=[]
        self.debug_log['cos_sim']=[]
        self.debug_log['aggr_weights']=[]
        self.debug_log['label_skew_ratio']=[]
        n_nets=num_of_benign_nets+num_of_mal_nets
        self.cos_matrices=[]
        #self.cos_matrix.append(np.zeros((n_nets, n_nets)))

    def cluster_grads(self, iter=-1):
        nets = self.benign_nets + self.mal_nets
        for net in nets:
            net.calc_grad(self.global_net.state_dict(), change_self=False)

        from sklearn.cluster import AgglomerativeClustering
        X = [np.array(net.grad_params) for net in nets]
        X= np.array(X)
        clustering = AgglomerativeClustering(n_clusters=num_of_distributions, affinity='cosine', linkage='complete').fit(X)
        from sklearn.metrics.cluster import adjusted_rand_score
        # print('Original Copylist', copylist)
        # print('Found clusters', clustering.labels_)



        #print('Original groups', [np.argwhere(np.array(copylist)==i).flatten() for i in range(num_of_distributions)])
        #print('Clustered groups', [np.argwhere(clustering.labels_==i).flatten() for i in range(num_of_distributions)])
        print('Clustering score', adjusted_rand_score(clustering.labels_.tolist(), copylist))
        self.log.append((iter, 'Original copylist', 'cluster_grads', copylist))
        self.log.append((iter, 'Clusters', 'cluster_grads', clustering.labels_))
        self.debug_log['cluster_without_running_avg'].append((iter, 'Cluster Score', 'cluster_grads', adjusted_rand_score(clustering.labels_.tolist(), copylist)))

        correct_c=0
        wrong_c=0
        for id, mal in enumerate(range(num_of_workers-num_of_mal_workers, num_of_workers)):
            mals_benign_brothers = np.where(np.array(copylist)==copylist[mal])
            mals_benign_brothers_clusters = [clustering.labels_[iid] for iid in mals_benign_brothers[0]]
            # print(mals_benign_brothers, mals_benign_brothers_clusters)
            benign_group_num=stats.mode(mals_benign_brothers_clusters)[0]

            if clustering.labels_[mal]==benign_group_num:
                correct_c += 1
            else:
                wrong_c += 1

        print('correct_c ', correct_c, 'wrong_c ', wrong_c)
        self.debug_log['cluster_mal_wra'].append((iter, 'correct_c', correct_c, 'wrong_c', wrong_c))

        
        coses=[]
        
        for i1, net1 in enumerate(nets):
            coses_l=[]
            for i2, net2 in enumerate(nets):
                coses_l.append(cos_calc_btn_grads(net1.grad_params, net2.grad_params))
            coses.append(coses_l)
            
        coses = np.array(coses)
        
        '''
        self.cos_matrix = self.cos_matrix + coses
        self.cos_matrix = self.cos_matrix/np.max(self.cos_matrix)
        '''
        self.cos_matrices.append(coses)
        print(len(self.cos_matrices))
        
        num_of_coses = np.minimum(len(self.cos_matrices), 5)
        
        cos_matrix=np.zeros((len(nets), len(nets)))
        for i in range(num_of_coses):
            cos_matrix+=self.cos_matrices[-i-1]
            
        cos_matrix = cos_matrix/num_of_coses
                    
        #print(cos_matrix)
        
        clustering = AgglomerativeClustering(n_clusters=num_of_distributions, affinity='precomputed', linkage='complete').fit(1-cos_matrix)
        # print('Original Copylist', copylist)
        # print('Found clusters', clustering.labels_)
        # print('Original groups', [np.argwhere(np.array(copylist)==i).flatten() for i in range(num_of_distributions)])
        # print('Clustered groups', [np.argwhere(clustering.labels_==i).flatten() for i in range(num_of_distributions)])
        print('\n\nClustering score with running avg', adjusted_rand_score(clustering.labels_.tolist(), copylist))
        self.debug_log['cluster'].append((iter, 'Cluster Score', 'cluster_grads', adjusted_rand_score(clustering.labels_.tolist(), copylist)))
        self.debug_log['cluster_labels'].append((iter, 'cluster_grads', 'Original cluster labels', copylist, 'Found cluster labels', clustering.labels_))
        

        correct_c=0
        wrong_c=0
        for id, mal in enumerate(range(num_of_workers-num_of_mal_workers, num_of_workers)):
            mals_benign_brothers = np.where(np.array(copylist)==copylist[mal])
            mals_benign_brothers_clusters = [clustering.labels_[iid] for iid in mals_benign_brothers[0]]
            # print(mals_benign_brothers, mals_benign_brothers_clusters)
            benign_group_num=stats.mode(mals_benign_brothers_clusters)[0]
            if clustering.labels_[mal]==benign_group_num:
                correct_c += 1
            else:
                wrong_c += 1
        print('correct_c ', correct_c, 'wrong_c ', wrong_c)
        self.debug_log['cluster_mal_wra'].append((iter, 'correct_c', correct_c, 'wrong_c', wrong_c))
        
        '''
        X = [np.array(net.grad_params) for net in self.benign_nets]
        X= np.array(X)
        copylist2=copylist[:self.num_of_benign_nets]
        clustering = AgglomerativeClustering(n_clusters=len(set(copylist2)), affinity='cosine', linkage='complete').fit(X)
        print('Original Copylist', copylist2)
        print('Found clusters', clustering.labels_)
        print('Original groups', [np.argwhere(np.array(copylist2)==i).flatten() for i in range(num_of_distributions)])
        print('Clustered groups', [np.argwhere(clustering.labels_==i).flatten() for i in range(num_of_distributions)])
        print('Clustering score', adjusted_rand_score(clustering.labels_.tolist(), copylist2))
        '''
        
        return coses

    def FLtrust(self, iter=-1):
        clean_server_grad=None
        grads=[]
        nets_grads=[]
        
        nets = self.benign_nets + self.mal_nets
        for net in nets:
            net.calc_grad(self.global_net.state_dict(), change_self=False)
            grad_net = CNN().to(device)
            grad_net.to(device)
            grad_net.copy_params(net.state_dict())
            grad_net.aggregate([self.global_net.state_dict()], aggr_weights=[-1])
            nets_grads.append(grad_net)

        for i in range(self.num_of_benign_nets):
            grads.append(self.benign_nets[i].grad_params)
            if i==self.num_of_benign_nets-1:
                clean_server_grad=grads[i]

        for i in range(self.num_of_mal_nets):
            grads.append(self.mal_nets[i].grad_params)
        
        norms = [torch.linalg.norm(grad) for grad in grads]
        #print('Norms of local gradients ', norms)
        self.log.append((iter, 'Norms of local gradients ', 'FLTrust', norms))

        
        cos_sims=[cos_calc_btn_grads(grad, clean_server_grad) for grad in grads]

        '''
        for grad in grads:
            cos_sims.append(torch.dot(grad, clean_server_grad)/ (torch.linalg.norm(grad)+ 1e-9) / (torch.linalg.norm(clean_server_grad)+ 1e-9))
        '''
        #print('\n Aggregating models')

        #print([cos_calc() ])

        #print('Cosine Similarities: ', cos_sims)
        self.log.append((iter, 'Cosine Similarities', 'FLtrust', cos_sims))
        self.debug_log['cos_sim'].append((iter, 'Cosine Similarities', 'FLtrust', cos_sims))
        cos_sims = np.maximum(np.array(cos_sims), 0)
        norm_weights = cos_sims/(np.sum(cos_sims)+1e-9)
        for i in range(len(norm_weights)):
            norm_weights[i] = norm_weights[i] * torch.linalg.norm(clean_server_grad) / (torch.linalg.norm(grads[i]))
        
        #print('Aggregation Weights: ', norm_weights)
        self.log.append((iter, 'Aggregation Weights', 'FLtrust', norm_weights))
        self.debug_log['aggr_weights'].append((iter, 'Aggregation Weights', 'FLtrust', norm_weights))

        self.global_net.aggregate([grad.state_dict() for grad in nets_grads], aggr_weights=norm_weights)
            
            

    def train_local_net(self, is_malicious, net_id, iter, ref_net_for_minimizing_dist=None, print_flag=False):
        if is_malicious:
            network=self.mal_nets[net_id]
            optim=self.mal_net_optims[net_id]
            # will change later to aggregate of benign_nets
            if ref_net_for_minimizing_dist is None:
                ref_net_for_minimizing_dist = self.benign_nets[0].parameters()
            ref_grad, ref_net = ref_net_for_minimizing_dist
            if ref_grad is None:
                import sys
                sys.exit()
            if ref_grad is not None:
                optim.__setrefgradparams__(ref_grad.parameters())
            optim.__setrefparams__(ref_net.parameters())
        else:
            network=self.benign_nets[net_id]
            optim=self.benign_net_optims[net_id]

        (_, _, trainloader) = train_loaders[iter][net_id + is_malicious*self.num_of_benign_nets]

        poisonNow = True if is_malicious and iter>=self.poison_starts_at_iter else False
        for epoch in range(self.n_epochs if not poisonNow else (self.n_epochs*1)):
            clientType = 'Malicious' if is_malicious else 'Benign'
            if print_flag:
                print(f'Iter {iter} - Epoch {epoch} - Client Type: {clientType} - Client Number {net_id} - Poison Training {poisonNow}')
            train_net(network, optim, trainloader, epoch, poisonNow=poisonNow, attack_type=self.attack_type)
        network.calc_grad(self.global_net.state_dict(), change_self=False)
        if poisonNow:
            test_label_flip(network)
        #     acc=test(network)
        #     self.log.append((iter, 'Local net test accuracy: mal', 'train_local_net', acc))
        #     if self.attack_type=='backdoorq':
        #         acc = backdoor_test(network)
        #         self.log.append((iter, 'Local net backdoor test accuracy: mal', 'train_local_net', acc))

    def train(self, tqdm_disable=False):
        for iter in range(self.n_iter):
            distanceList=[]
            cosList=[]
            networks=[]
            networks+=self.benign_nets
            networks+=self.mal_nets

            for i in tqdm(range(self.num_of_benign_nets), disable=tqdm_disable):
                self.train_local_net(False, i, iter)


            benign_aggr_net=CNN().to(device)
            benign_aggr_net.set_param_to_zero()

            ### if adversary knows benign_net_aggregates
            benign_aggr_net.aggregate([net.state_dict() for net in self.benign_nets])
            ### if adversary knows clean server
            #benign_aggr_net.copy_params(self.benign_nets[-1].state_dict())

            benign_aggr_net_grad=CNN().to(device)
            benign_aggr_net_grad.copy_params(benign_aggr_net.state_dict())
            benign_aggr_net_grad.aggregate([self.global_net.state_dict()], aggr_weights=[-1])

            
            for i in tqdm(range(self.num_of_mal_nets), disable=True):
                self.train_local_net(True, i, iter, ref_net_for_minimizing_dist=(benign_aggr_net_grad, benign_aggr_net))
                
                if self.scale_up:
                    scaled_up_grad = get_scaled_up_grads(self.global_net, networks, self, iter)
                    self.mal_nets[i].copy_params(scaled_up_grad.state_dict())
                    #self.mal_nets[i].aggregate([benign_aggr_net.state_dict()])
                

            coses = self.cluster_grads(iter)

            self.debug_log['coses'].append((iter, coses))

            cosList=[cos_calc_btn_grads(net.grad_params, self.benign_nets[-1].grad_params) for net in networks]
            distanceList=[calcDiff(net, self.benign_nets[-1]) for net in networks]

            #self.cluster_grads()

            self.log.append((iter, 'Benign net distance', 'train', distanceList[:self.num_of_benign_nets]))
            #print('Benign net distance', distanceList[:self.num_of_benign_nets])
            self.log.append((iter, 'Malicious net distance', 'train', distanceList[self.num_of_benign_nets:]))
            #print('Malicious net distance', distanceList[self.num_of_benign_nets:])
            self.log.append((iter, 'Cos sim list', 'train', cosList))
            #print('cos_sim list ', cosList)

            # aggregate nets
            #self.global_net.set_param_to_zero()
            #self.global_net.aggregate([network.state_dict() for network in networks])
            self.FLtrust(iter=iter)
            print('\n\n\nAggregate test at iter ', iter)
            acc=test(self.global_net)
            self.log.append((iter, 'Test accuracy: agg net', 'train', acc))

            acc=test_label_flip(self.global_net, print_flag=True)
            self.debug_log['target_label_acc'].append((iter, 'Target label accuracy: agg net', 'train', acc))
            #backdoor_test(self.global_net)
            #self.log.append((iter, 'Backdoor test accuracy: agg net', 'train', acc))
            self.log.append((iter, 'Distance between aggregate global and clean server', 'train', calcDiff(self.global_net, self.benign_nets[-1])))

            # set all local nets equal to global net at the end of the iteration
            
            for network in networks:
                network.copy_params(self.global_net.state_dict())
            