import torch 
import torch.nn as nn 
import numpy as np

## Set up basic unet architecture to estimate the score function 

class scoreNet(nn.Module):
    """
    Custom neural network, used to estimate the score function, which is needed in the sampling process of the g
    generative model.
    Architecture based on the U-net [O.Ronneberger et al.]
    """
    def __init__(self):
        super().__init__()
        #attributes needed

    def double_conv(channel_in,channel_out):
        """
        Method to implement the double convolution step (as outlined in O.Renneberger et al.)
        Between each convolutional layer a ReLU activation is used 
        :param channel_in: number of channels on the input
        :param channel_out: number of output channels
        :return double_convolution: the output of the double convolution process 
        """
        double_convolution = nn.Sequential(nn.Conv2d(channel_in,channel_out,kernel_size=3),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(channel_out,channel_out,kernel_size=3),
                                           nn.ReLU(inplace=True))
        return double_convolution