import torch 
import torch.nn as nn 
import numpy as np


## Set up basic unet architecture to estimate the score function 
#test comment

class GaussianFourierProjection(nn.Module):
    def __init__(self,embed_dim,scale=30.):
        """

        """
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2)*scale, requires_grad=False)
    def forward(self,x):
        """
        method for the forward process
        :param x: 
        """
        x_proj = x[:,None] * self.W[None,:]*2*np.pi
        return torch.cat([torch.sin(x_proj),torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """Fully connected layer reshaping outputs to feature maps"""
    def __init__(self,input_dim,output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim,output_dim)
    def forward(self, x):
        return self.dense(x)[...,None,None]
    
    
class scoreNet(nn.Module):
    """
    Custom neural network, used to estimate the score function, which is needed in the sampling process of the g
    generative model.
    Architecture based on the U-net [O.Ronneberger et al.]
    """
    def __init__(self,marginal_prob_std,embed_dim=256):
        super().__init__()
        #attributes needed
        self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),nn.Linear(embed_dim, embed_dim))
        self.activation = nn.ReLU(inplace=True)
        self.max_pool = self.max_pooling_downsample()
        self.down_conv_1 = self.double_conv(1,32,2)
        self.dense1 = Dense(embed_dim, 32)
        self.down_conv_2 = self.double_conv(32,64,2)
        self.dense2 = Dense(embed_dim, 32)
        self.down_conv_3 = self.double_conv(64,128,2)
        self.dense3 = Dense(embed_dim, 64)
        
        self.up_conv1 = self.double_conv_up(128,64,2)
        self.dense4 = Dense(embed_dim, 128)
        self.up_process_1 = nn.ConvTranspose2d(in_channels=64,out_channels=64,kernel_size=2,stride=2,output_padding=1)
        self.up_conv2 = self.double_conv(128,32,kernel_size=3)
        self.dense5 = Dense(embed_dim, 32)
        self.up_process_2 = nn.ConvTranspose2d(in_channels=32,out_channels=32,kernel_size=2, stride=2)
        self.up_conv3 = self.double_conv_up(64,1,6)
        self.output = nn.Conv2d(in_channels=16,out_channels=1, kernel_size=1)

        

    def double_conv(self,channel_in,channel_out,kernel_size=2):
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
    
    def max_pooling_downsample(self,kernel_size=2,stride=2):
        """
        we use max_pooling in the downsampling 
        :param kernel_size: size of the kernel which performs max pooling
        :param stride: stride of the kernel used in the max pooling process
        """
        max_pool = nn.MaxPool2d(kernel_size,stride)
        return max_pool
    
    def double_conv_up(self,channel_in, channel_out,kernel_size=3):
        """
        Method to implement a double convolution however the image dims increase by 2 in each dimension
        :param channel_in: number of in channels 
        :param channel_out: number of out channels
        """
        double_conv_up = nn.Sequential(nn.ConvTranspose2d(channel_in,channel_out,kernel_size,stride=1),
                                       nn.ReLU(inplace=True),
                                       nn.ConvTranspose2d(channel_out,channel_out,kernel_size,stride=1),
                                       nn.ReLU(inplace=True))
        return double_conv_up

    def crop(self,input,target):
        """
        Method to implement the crop process (as outlined in O.Renneberger et al.)
        ***************** function from tutorial ****************
        :param input:
        :param target: 
        """
        target_size = target.size()[2]
        input_size = input.size()[2]
        diff = (input_size - target_size)//2
        return input[:,:,diff:input_size-diff, diff:input_size-diff]
    
    
    
    def forward(self,x,t):
        """
        Forward pass of the neural network
        :param x: the input data point 
        """
        #encoding process
        print("the shape at the start is x1", x.size())
        embedding = self.activation(self.embed(t))
        x1 = self.down_conv_1(x)#concat
        print(x1.size())
        x1_temp= self.dense1(embedding)
        print(x1_temp.size())
        x2 = self.max_pool(x1)
        print(x2.size())
        x2_temp = self.dense2(embedding)
        print(x2_temp.size())
        x3 = self.down_conv_2(x2)#concat
        print("This is the shape of x3", x3.size())
        x4 = self.max_pool(x3)
        x4 += self.dense3(embedding)
        x5 = self.down_conv_3(x4)
        

        #decode process
        x6 = self.up_conv1(x5)
        print("this is the size of x6",x6.size())
        x6_temp = self.dense4(embedding)
        print("THIS",x6_temp.size())
        x7 = self.up_process_1(x6)
        unet_crop = self.crop(x3,x7)
        intermed = torch.cat([x7,unet_crop],1)
        x7 = self.up_conv2(intermed)
        
        x7 = self.up_process_2(x7)
        x7 += self.dense5(embedding)
        unet_crop = self.crop(x1,x7)
        x7 = self.up_conv3(torch.cat([x7,unet_crop],1))
        print(x7.size())
        #!!normalise output
        return x7


def marginal_forward(time,sigma=torch.Tensor([25]),std=False):
    """
    We use this function to perturb the data to various time steps
    :param time: the time step to perturb to (numpy tensor)
    :param sigma: the diffusion coefficent
    :param std (optional): return the standard deviation 
    """
    marginal_forward_val = 1/(2*torch.log(sigma))*(sigma**(2*time)-1)
    marginal_forward_val_std = torch.sqrt(marginal_forward_val)
    if std:
        return marginal_forward_val_std
    else:
        return marginal_forward_val
random_time = torch.rand(1)
img = torch.rand((1,1,28,28))
testnet = scoreNet(marginal_forward)
testnet.forward(img,random_time)

