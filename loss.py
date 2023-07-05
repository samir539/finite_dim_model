import numpy as np
import torch 
import torch.nn as nn


#forward process
################
def marginal_forward(time,sigma,std=False):
    """
    We use this function to perturb the data to various time steps
    :param time: the time step to perturb to (numpy tensor)
    :param sigma: the diffusion coefficent
    :param std (optional): return the standard deviation 
    """
    marginal_forward_val = 1/(2*torch.log(sigma))*(sigma**(2*time)-1)
    marginal_forward_val_std = torch.std(marginal_forward_val)
    if std:
        return marginal_forward_val_std
    else:
        return marginal_forward_val
        







## loss function
################

def loss_function(x, marginal_forward):
    """
    Loss function to pass into 
    :param x: 
    :param marginal_forward: 
    """

    #generate a random time
    random_time = torch.rand(1)


    #make random tensor from gaussian
    random_like = torch.randn_like(x)



    #get marginal prob std
    
    

    #perturb data

    #use score net to get score


#compute loss