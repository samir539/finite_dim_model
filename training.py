import torch 
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision


def load_data(batch_size,data_set):
    """
    Make instance of dataloader class
    :param batch_size: samples in a backward and forward pass
    :param data_set: data set
    """
    data_loader = DataLoader(data_set=data_set, batch_size=batch_size, shuffle=True)
    return data_loader
    
        

MNIST = torchvision.datasets.MNIST(root="./",download=True,train=True,transform=transforms.Compose([transforms.ToTensor()]),)

