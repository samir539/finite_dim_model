import numpy as np
import torch 
import torch.nn as nn
from score_net import scoreNet
torch.manual_seed(0)

#forward process
################
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
    
    
## diffusion coef
def diffusion_coeff(t, sigma=25):
  """Compute the diffusion coefficient of our SDE.

  Args:
    t: A vector of time steps.
    sigma: The $\sigma$ in our SDE.
  
  Returns:
    The vector of diffusion coefficients.
  """
  return torch.tensor(sigma**t)
        

## loss function
################

def loss_function(net,x, marginal_forward,sigma=25):
    """
    Loss function to pass into 
    :param net: the network which learns the score
    :param x: 
    :param marginal_forward: 
    """

    #generate a random time
    random_time = torch.rand(1)
    #make random tensor from gaussian
    random_like = torch.randn_like(x)
    #get marginal prob std
    marginal_prob_std = marginal_forward(random_time,std=True)
    #perturb data
    x_perturbed = x + random_like*marginal_prob_std
    #use score net to get score
    score_model = net
    score_estimate = score_model.forward(x_perturbed)
    #compute loss
    loss = (1/2)*torch.mean(torch.sum((score_estimate + (x_perturbed - x)/sigma**2)**2, dim=(1,2,3)))
    # loss = torch.mean(torch.sum((score_estimate * marginal_prob_std[:, None, None, None] + random_like)**2, dim=(1,2,3)))
    return loss


# img = torch.rand((1,1,28,28))
# print(loss_function(img,marginal_forward))