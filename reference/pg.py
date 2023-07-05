import numpy as np
import torch 
import torch.nn as nn
import torch
import functools
from torch.optim import Adam
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from tqdm import tqdm
from score_based_model import ScoreNet
from forward_sde import marginal_prob_std_fn
from loss_function import loss_fn
from matplotlib import pyplot as plt

device = 'cpu' #@param ['cuda', 'cpu'] {'type':'string'}

# class TestClass:
#     def __init__(self):
#         self.w = nn.Parameter(torch.randn(256//2)*30.0, requires_grad=False)
    

# class GaussianFourierProjection(nn.Module):
#     def __init__(self,embed_dim,scale=30.):
#         """
#         """
#         super().__init__()
#         self.W = nn.Parameter(torch.randn(embed_dim // 2)*scale, requires_grad=False)
#     def forward(self,x):
#         """
#         method for the forward process
#         :param x: 
#         """
#         print("this is self.w", self.W)
#         x_proj = x[:,None] * self.W[None,:]*2*np.pi
#         return torch.cat([torch.sin(x_proj),torch.cos(x_proj)], dim=-1)

# embed_dim = 4
# embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
#          nn.Linear(embed_dim, embed_dim))



# W = nn.Parameter(torch.randn(8 // 2)*1, requires_grad=False)
# print(W)
# test_object  = GaussianFourierProjection(4)
# print(test_object.W)
# print(test_object.W.shape)
# x_proj = torch.tensor([[0,0],[0,0]])
# val1  = torch.sin(x_proj)
# val2 = torch.cos(x_proj)
# print(val1)
# print(val2)
# val = torch.cat([torch.sin(x_proj),torch.cos(x_proj)], dim=-1)
# print(val)


# dataset = MNIST('.', train=True, transform=transforms.ToTensor(), download=True)
# data_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
# train_feat, train_label = next(iter(data_loader))
# img = train_feat.squeeze()
# plt.imshow(img)
# plt.show()
# print(train_feat.shape)

# for x,y in enumerate(data_loader):
#     if x == 27:
#         data = y[0]
#         target = y[1]
#         break
# #
# print(data.size())
# img = data.squeeze()
# plt.imshow(img)
# plt.show()
# print(target)

# for x,y in data_loader:
# #     print(x.shape)

# embed_dim = 16
# class GaussianFourierProjection(nn.Module):
#     def __init__(self,embed_dim,scale=30.):
#         """
#         """
#         super().__init__()
#         self.W = nn.Parameter(torch.randn(embed_dim // 2)*scale, requires_grad=False)
#         # self.W = nn.Parameter(0.2)
#     def forward(self,x):
#         """
#         method for the forward process
#         :param x: 
#         """
#         x_proj = x[:,None] * self.W[None,:]*2*np.pi
#         return torch.cat([torch.sin(x_proj),torch.cos(x_proj)], dim=-1)
    
# act = lambda x: x * torch.sigmoid(x)
# embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
#          nn.Linear(embed_dim, embed_dim))

# # print(act(embed(0.5)))
# eps = 1e-5
# test_gaus_proj = GaussianFourierProjection(8)
# time  = torch.rand(1) * (1. - eps) + eps  #random value from [0,1) unif
# # time = torch.tensor([0.5])
# # print(time)
# val = test_gaus_proj(time)

# print(val)

# valval = torch.randn(8 // 2)
# print(valval)

# import torch
# import torch.nn as nn

# # Define a convolutional layer with GroupNorm
# conv = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3)
# norm = nn.GroupNorm(num_groups=4, num_channels=32)

# # Generate some dummy input
# x = torch.randn(10, 16, 28, 28)  # Batch of 10 images, each with 16 channels

# # Pass the input through the convolutional layer
# y = conv(x)
# print(y.shape)

# # Apply GroupNorm to the output
# y_norm = norm(y)

# # Print the shape of the normalized output
# print(y_norm.shape)


# import torch
# import torch.nn as nn

# # Define an input tensor
# input_tensor = torch.randn(2, 6, 4, 4)  # Batch size of 2, 6 channels, spatial size of 4x4

# # Apply GroupNorm with 2 groups
# group_norm = nn.GroupNorm(num_groups=2, num_channels=6)
# output_tensor = group_norm(input_tensor)

# # Print the shapes of the input and output tensors
# print("Input shape:", input_tensor.shape)
# print("Output shape:", output_tensor.shape)
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
    


class ScoreNet(nn.Module):
  """A time-dependent score-based model built upon U-Net architecture."""

  def __init__(self, marginal_prob_std, channels=[32, 64, 128, 256], embed_dim=256):
    """Initialize a time-dependent score-based network.

    Args:
      marginal_prob_std: A function that takes time t and gives the standard
        deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
      channels: The number of channels for feature maps of each resolution.
      embed_dim: The dimensionality of Gaussian random feature embeddings.
    """
    super().__init__()
    # Gaussian random feature embedding layer for time
    self.embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
         nn.Linear(embed_dim, embed_dim))
    # Encoding layers where the resolution decreases
    self.conv1 = nn.Conv2d(1, channels[0], 3, stride=1, bias=False)
    self.dense1 = Dense(embed_dim, channels[0])
    self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
    self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
    self.dense2 = Dense(embed_dim, channels[1])
    self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
    self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
    self.dense3 = Dense(embed_dim, channels[2])
    self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
    self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
    self.dense4 = Dense(embed_dim, channels[3])
    self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])    

    # Decoding layers where the resolution increases
    self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False)
    self.dense5 = Dense(embed_dim, channels[2])
    self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
    self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)    
    self.dense6 = Dense(embed_dim, channels[1])
    self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
    self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)    
    self.dense7 = Dense(embed_dim, channels[0])
    self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
    self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)
    
    # The swish activation function
    self.act = lambda x: x * torch.sigmoid(x)
    self.marginal_prob_std = marginal_prob_std
  
  def forward(self, x, t): 
    # Obtain the Gaussian random feature embedding for t   
    print("the size of x is ", x.size())
    embed = self.act(self.embed(t))    

    # Encoding path
    h1 = self.conv1(x)  
    print("the size of h1 is ",h1.size())
    ## Incorporate information from t
    h1 += self.dense1(embed)
    print("the size of h1 after dense is ",h1.size())
    ## Group normalization
    h1 = self.gnorm1(h1)
    print("the size of h1 after gnorm is ",h1.size())
    h1 = self.act(h1)
    print("the size of h1 after act is  ",h1.size())

    h2 = self.conv2(h1)
    print("the size of h2 is ",h2.size())
    h2 += self.dense2(embed)
    h2 = self.gnorm2(h2)
    h2 = self.act(h2)
    print("the size of h2 post activation is  ",h2.size())
    h3 = self.conv3(h2)
    print("the size of h3 is  ",h3.size())
    h3 += self.dense3(embed)
    h3 = self.gnorm3(h3)
    h3 = self.act(h3)

    
    h4 = self.conv4(h3)
    print("the size of h4  is  ",h4.size())
    h4 += self.dense4(embed)
    h4 = self.gnorm4(h4)
    h4 = self.act(h4)
    print("the size of h4  post activation is   ",h4.size())

    # Decoding path
    h = self.tconv4(h4)
    print("h size,", h.size())
    ## Skip connection from the encoding path 
    h += self.dense5(embed)
    h = self.tgnorm4(h)
    h = self.act(h)
    print("h size,", h.size())
    h = self.tconv3(torch.cat([h, h3], dim=1))
    print("h size tconv3,", h.size())
    h += self.dense6(embed)
    h = self.tgnorm3(h)
    h = self.act(h)
    print("h size tconv3 act,", h.size())
    print("the size of h is ", h.size(), "the size of h2 is", h2.size())
    h = self.tconv2(torch.cat([h, h2], dim=1))
    print("h size tconv4 act,", h.size())
    h += self.dense7(embed)
    h = self.tgnorm2(h)
    h = self.act(h)
    h = self.tconv1(torch.cat([h, h1], dim=1))

    # Normalize output
    print("this is the size of h", h.size())
    h = h / self.marginal_prob_std(t)[:, None, None, None]
    print("this is the size of h", h.size())
    return h

img = torch.rand((1,32,15,15))
test_img =  torch.rand((1,64,11,11))
# testnet = ScoreNet(marginal_prob_std_fn)
# testnet(test_img,torch.Tensor([0.5]))


def crop(img,target_img):
        target_size = target_img.size()[2]
        tensor_size = img.size()[2]
        delta = tensor_size - target_size
        delta = delta // 2
        return img[:,:, delta:tensor_size-delta, delta:tensor_size-delta]



out = crop(test_img,img)
print(out.size())
crop_out = torch.cat([out, img])
print(crop_out.size())
# print(img.size())
# out = out.squeeze()
# plt.imshow(out)
# plt.show()

import torch 
import torch.nn as nn 

def double_convolution(in_channel,out_channel):
    """
    function to implement a sequential double convolution where each convolution is followed by a ReLU activation
    """
    conv = nn.Sequential(nn.Conv2d(in_channel,out_channel, kernel_size=3),
                         nn.ReLU(inplace=True),
                         nn.Conv2d(out_channel,out_channel, kernel_size=3),
                         nn.ReLU(inplace=True))
    return conv    

def crop(img,target_img):
        target_size = target_img.size()[2]
        tensor_size = img.size()[2]
        delta = tensor_size - target_size
        delta = delta // 2
        return img[:,:, delta:tensor_size-delta, delta:tensor_size-delta] 

class Unet(nn.Module):
    """

    """
    def __init__(self):
        super().__init__()
        self.max_pool_2by2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.down_conv_1 = double_convolution(1,64)
        self.down_conv_2 = double_convolution(64,128)
        self.down_conv_3 = double_convolution(128,256)
        self.down_conv_4 = double_convolution(256,512)
        self.down_conv_5 = double_convolution(512,1024)

        self.up_process_1 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv_1 = double_convolution(1024,512)

        self.up_process_2 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv_2 = double_convolution(512,256)

        self.up_process_3 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv_3 = double_convolution(256,128)

        self.up_process_4 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.up_conv_4 = double_convolution(128,64)

        self.output_layer = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1)


    def forward(self, image):
        """
        take an image and run the forward pass
        """

        #encoding process
        x1 = self.down_conv_1(image) #concatenate
        x2 = self.max_pool_2by2(x1)
        x3 = self.down_conv_2(x2) # concatenate
        x4 = self.max_pool_2by2(x3)
        x5 = self.down_conv_3(x4) # concatenate
        x6 = self.max_pool_2by2(x5)
        x7 = self.down_conv_4(x6) # concatenate
        x8 = self.max_pool_2by2(x7)
        x9 = self.down_conv_5(x8) 
        

        #decoding process
        x10 = self.up_process_1(x9)
        y = crop(x7,x10)
        x10 = self.up_conv_1(torch.cat([x10,y],1))
        
        x10 = self.up_process_2(x10)
        y = crop(x5,x10)
        x10 = self.up_conv_2(torch.cat([x10,y],1))

        x10 = self.up_process_3(x10)
        y = crop(x3,x10)
        x10 = self.up_conv_3(torch.cat([x10,y],1))

        x10 = self.up_process_4(x10)
        y = crop(x1,x10)
        x10 = self.up_conv_4(torch.cat([x10,y],1))
        print(x10.size())

        #output channel
        x10 = self.output_layer(x10)
        print(x10.size())
        # print(x10)
        return x10



        
      
if __name__ == "__main__":
    image = torch.rand((1,1,572,572))
    model = Unet()
    # print(model.forward(image))

