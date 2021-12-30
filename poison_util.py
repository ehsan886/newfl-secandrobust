import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np

import copy

from params import *


def add_pixel_pattern(ori_image):
    image = copy.deepcopy(ori_image)
    poison_patterns= poison_dict['poison_pattern']
    delta =  poison_dict['poison_delta']

    for i in range(0, len(poison_patterns)):
        pos = poison_patterns[i]
        image[0][pos[0]][pos[1]] = min( image[0][pos[0]][pos[1]] + delta/np.sqrt(len(poison_patterns)), 1)


    return image

def get_poison_batch(bptt,adversarial_index=-1, evaluation=False, attack_type='label_flip'):
    if attack_type=='backdoor':
        import sys
        print("still backdooring")
        sys.exit()

        images, targets = bptt

        poison_count= 0
        new_images=images
        new_targets=targets

        for index in range(0, len(images)):
            if evaluation: # poison all data when testing
                new_targets[index] = poison_dict['poison_label_swap']
                new_images[index] = add_pixel_pattern(images[index])
                poison_count+=1

            else: # poison part of data when training
                if index < poison_dict['poisoning_per_batch']:
                    new_targets[index] = poison_dict['poison_label_swap']
                    new_images[index] = add_pixel_pattern(images[index])
                    poison_count += 1
                else:
                    new_images[index] = images[index]
                    new_targets[index]= targets[index]

        new_images = new_images.to(device)
        new_targets = new_targets.to(device).long()
        if evaluation:
            new_images.requires_grad_(False)
            new_targets.requires_grad_(False)
        return new_images,new_targets,poison_count
    
    elif attack_type=='degrade' or attack_type=='label_flip':
        images, targets = bptt

        poison_count= 0
        new_images=images
        new_targets=targets

        num_of_classes = 10

        for index in range(0, len(images)):
            if evaluation: # poison all data when testing
                if attack_type=='degrade':
                    new_targets[index] = poison_dict['poison_label_swap']
                elif attack_type=='label_flip':
                    new_targets[index] = (targets[index]+1)%num_of_classes
                new_images[index] = images[index]
                poison_count+=1

            else: # poison part of data when training
                if index < poison_dict['poisoning_per_batch']:
                    new_targets[index] = poison_dict['poison_label_swap']
                    new_images[index] = add_pixel_pattern(images[index])
                    poison_count += 1
                else:
                    new_images[index] = images[index]
                    new_targets[index]= targets[index]

        new_images = new_images.to(device)
        new_targets = new_targets.to(device).long()
        if evaluation:
            new_images.requires_grad_(False)
            new_targets.requires_grad_(False)
        return new_images,new_targets,poison_count


def get_poison_batch_special_label_flip(bptt, num_of_classes=10, target_class=0):
    images, targets = bptt

    poison_count= 0
    new_images=images
    new_targets=targets

    for index in range(0, len(images)):
        if targets[index]==target_class: # poison all data when testing
            new_targets[index] = num_of_classes-target_class-1
            new_images[index] = images[index]
            poison_count+=1

        else: 
            new_images[index] = images[index]
            new_targets[index]= targets[index]

    new_images = new_images.to(device)
    new_targets = new_targets.to(device).long()

    return new_images,new_targets,poison_count
        

def get_batch(bptt, evaluation=False):
    data, target = bptt
    data = data.to(device)
    target = target.to(device)
    if evaluation:
        data.requires_grad_(False)
        target.requires_grad_(False)
    return data, target