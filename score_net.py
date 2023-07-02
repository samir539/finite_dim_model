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

    def double_conv(channel_in,channel_out,kernel_size):
        """
        Method to implement the double convolution step (as outlined in O.Renneberger et al.)
        Between each convolutional layer a ReLU activation is used 
        Image dim decreases by 2 in each dimension
        :param channel_in: number of channels on the input
        :param channel_out: number of output channels
        :kernel_size: the kernel size
        :return double_convolution: the output of the double convolution process 
        """
        double_convolution = nn.Sequential(nn.Conv2d(channel_in,channel_out,kernel_size=2),
                                           nn.ReLU(inplace=True),
                                           nn.Conv2d(channel_out,channel_out,kernel_size=2),
                                           nn.ReLU(inplace=True))
        return double_convolution
    
    def max_pooling_downsample(kernel_size=2,stride=2):
        """
        we use max_pooling in the downsampling 
        :param kernel_size: size of the kernel which performs max pooling
        :param stride: stride of the kernel used in the max pooling process
        """
        max_pool = nn.MaxPool2d(kernel_size,stride)
        return max_pool
    
    def double_conv_up(channel_in, channel_out):
        """
        Method to implement a double convolution however the image dims increase by 2 in each dimension
        :param channel_in: number of in channels 
        :param channel_out: number of out channels
        """
        double_conv_up = nn.Sequential(nn.ConvTranspose2d(channel_in,channel_out,3,stride=1),
                                       nn.ReLU(inplace=True),
                                       nn.ConvTranspose2d(channel_out,channel_out,3,stride=1),
                                       nn.ReLU(inplace=True))
        return double_conv_up



