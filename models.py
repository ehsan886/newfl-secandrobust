import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import numpy as np

from util import convert_model_to_param_list

class SimpleNet(nn.Module):
    def __init__(self, name=None, created_time=None, is_malicious=False, net_id=-1):
        super(SimpleNet, self).__init__()
        self.created_time = created_time
        self.name=name

        self.is_malicious=is_malicious
        self.net_id=net_id


    def save_stats(self, epoch, loss, acc):
        self.stats['epoch'].append(epoch)
        self.stats['loss'].append(loss)
        self.stats['acc'].append(acc)

    def copy_params(self, state_dict, coefficient_transfer=100):

        own_state = self.state_dict()

        for name, param in state_dict.items():
            if name in own_state:
                shape = param.shape
                own_state[name].copy_(param.clone())
    
    def set_param_to_zero(self):
        own_state = self.state_dict()

        for name, param in own_state.items():
            shape = param.shape
            param.mul_(0)       

    def aggregate(self, state_dicts, aggr_weights=None):
        #self.copy_params(state_dicts[0])
        own_state = self.state_dict()
        
        nw = len(state_dicts)
        if aggr_weights is None:
            aggr_weights = [1/nw]*nw

        for i, state_dict in enumerate(state_dicts):
            for name, param in state_dict.items():
                if name in own_state:
                    shape = param.shape
                    own_state[name].add_(param.clone().mul_(aggr_weights[i]))

    def calc_grad(self, state_dict, change_self=True):
        if change_self:
            own_state = self.state_dict()

            for name, param in state_dict.items():
                if name in own_state:
                    shape = param.shape
                    own_state[name].sub_(param.clone())
        else:
            self_params = convert_model_to_param_list(self)
            ref_params = convert_model_to_param_list(state_dict)

            self_params.sub_(ref_params)
            self.grad_params = self_params


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class MnistNet(SimpleNet):#model for mnist
    def __init__(self, name=None, created_time=None,num_of_classes = 10):
        super(MnistNet, self).__init__(f'{name}_Simple', created_time)
     
        self.fc_layer = torch.nn.Sequential(#1 * 28 * 28
            Flatten(),#784
            nn.Linear(784, num_of_classes),
        )
    
    def forward(self, x):
     
        out = self.fc_layer(x)
        return out


class FLNet(SimpleNet):
    def __init__(self, name=None, created_time=None,num_of_classes = 10):
        super(FLNet, self).__init__(f'{name}_Simple', created_time)

        self.mnistnet1 = MnistNet()

        self.mnistnet2 = MnistNet()

        self.mnistnetavg = MnistNet()

    def forward(self, x):
        out = self.mnistnet1(x)
        return out



class CNN(SimpleNet):
    def __init__(self, name=None, created_time=None,num_of_classes = 10,network_id=-1, net_id=-1, is_malicious=False):
        super(CNN, self).__init__(f'{name}_Simple', created_time, is_malicious=is_malicious, net_id=net_id)

        self.network_id=network_id

        self.conv1 = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),    
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(16, 32, 5, 1, 2),     
            nn.ReLU(),                      
            nn.MaxPool2d(2),                
        )
        # fully connected layer, output 10 classes
        self.out = nn.Linear(32 * 7 * 7, 10)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)       
        output = self.out(x)
        return output    # return x for visualization
'''
class CNN(SimpleNet):
    def __init__(self, name=None, created_time=None,num_of_classes = 10,network_id=-1, net_id=-1, is_malicious=False):
        super(CNN, self).__init__(f'{name}_Simple', created_time, is_malicious=is_malicious, net_id=net_id)
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
'''
