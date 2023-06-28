import numpy as np
import torch 
import torch.nn as nn

class TestClass:
    def __init__(self):
        self.w = nn.Parameter(torch.randn(256//2)*30.0, requires_grad=False)
    

test_object  = TestClass()
print(test_object.w)
print(test_object.w.shape)