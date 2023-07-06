import torch 
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision
import torch.optim as optim
from tqdm import tqdm
from score_net import scoreNet


MNIST = torchvision.datasets.MNIST(root="./",download=True,train=True,transform=transforms.Compose([transforms.ToTensor()]),)
def load_data(batch_size,data_set):
    """
    Make instance of dataloader class
    :param batch_size: samples in a backward and forward pass
    :param data_set: data set
    """
    data_loader = DataLoader(data_set=data_set, batch_size=batch_size, shuffle=True)
    return data_loader
    
    
def train_epoch(trainloader, net , optimiser,marginal_std, loss_function):
    """
    Function to train a single epoch
    :param trainloader: dataloader with training data
    :param net: the score net
    :param optimiser: the optimiser to use
    :param loss_function: the loss function
    
    :return average_loss: average loss 
    """
    total_loss = 0.0
    count = 0
    
    for index, data in enumerate(trainloader):
        inputs, labels = data

        optimiser.zero_grad()
        
        #forward pass
        output = net.forward(inputs)
        
        #loss
        loss = loss_function(net,inputs,marginal_std)
        loss.backwards()
        
        #update weights
        optimiser.step()
        
        #update loss
        total_loss += loss.item()*inputs.size(0)
        count += inputs.size(0)
        
    average_loss = total_loss/count
    return average_loss,count
    

        

def run_train(batch_size,learning_rate,epoch_num,loss_function, optimiser_choice, dataset_choice,marginal_std_fn):
    """
    Function to wrap score nn training 
    :param batch_size: the size of the batch
    :param learning_rate: the learning rate
    :param epoch_num: the number of epochs to run
    :param loss_function: the loss function
    :param optimiser_choice: the choice of optimiser (ADAM or SGD)
    :param dataset_choice: the dataset 
    """
    #init scoreNet
    score_net = scoreNet()
    
    #optimiser
    if optimiser_choice == 'SGD':
        optimiser = optim.SGD(score_net.parameters(), learning_rate, momentum=0.8)
    if optimiser_choice == 'ADAM':
        optimiser = optim.Adam(score_net.parameters(), learning_rate)
        
        
    data_loader = load_data(batch_size,dataset_choice)
    
    for i in tqdm(range(epoch_num)):
        mean_loss = 0
        count = 0
        avg_loss , count_epoch = train_epoch(data_loader,score_net,optimiser,marginal_std_fn,loss_function)
        mean_loss += avg_loss
        count += count_epoch
    
        torch.save(score_net.state_dict(), 'save.pth')
        
    return None
        
    
    