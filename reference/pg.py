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
#     print(x.shape)

embed_dim = 16
class GaussianFourierProjection(nn.Module):
    def __init__(self,embed_dim,scale=30.):
        """
        """
        super().__init__()
        self.W = nn.Parameter(torch.randn(embed_dim // 2)*scale, requires_grad=False)
        # self.W = nn.Parameter(0.2)
    def forward(self,x):
        """
        method for the forward process
        :param x: 
        """
        x_proj = x[:,None] * self.W[None,:]*2*np.pi
        return torch.cat([torch.sin(x_proj),torch.cos(x_proj)], dim=-1)
    
act = lambda x: x * torch.sigmoid(x)
embed = nn.Sequential(GaussianFourierProjection(embed_dim=embed_dim),
         nn.Linear(embed_dim, embed_dim))

# print(act(embed(0.5)))
eps = 1e-5
test_gaus_proj = GaussianFourierProjection(8)
time  = torch.rand(1) * (1. - eps) + eps  #random value from [0,1) unif
# time = torch.tensor([0.5])
# print(time)
val = test_gaus_proj(time)

print(val)

# valval = torch.randn(8 // 2)
# print(valval)