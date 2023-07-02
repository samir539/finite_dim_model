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
        self.max_pool = self.max_pooling_downsample()
        self.down_conv_1 = self.double_conv(1,32,2)
        self.down_conv_2 = self.double_conv(32,64,2)
        self.down_conv_3 = self.double_conv(64,128,2)
        
        self.up_conv1 = self.double_conv_up(128,64)
        self.up_process_1 = nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=2,stride=2,output_padding=1)
        self.up_conv2 = self.double_conv(64,32,kernel_size=3)
        self.up_process_2 = nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=2, stride=2)
        self.up_conv3 = self.double_conv(32,16)
        self.output = nn.Conv2d(in_channels=16,out_channels=1, kernel_size=1)

        

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

    def crop(input,target):
        """
        Method to implement the crop process (as outlined in O.Renneberger et al.)
        ***************** function from tutorial ****************
        :param input:
        :param target: 
        """
        target_size = target.size()[2]
        input_size = input.size()[2]
        diff = (input_size - target_size)//2
        return input[:,:,diff:target_size-diff, diff:target_size-diff]
    


