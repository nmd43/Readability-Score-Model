import torch
import torch.nn as nn
import math

class MLP(nn.Module):
    def __init__(self, data_shape, hidden_size, scale_factor, num_layers, activation, target_size):
        super().__init__()
        input_size = math.prod(data_shape)
        blocks = []
        for _ in range(num_layers):
            blocks.append(nn.Linear(input_size, hidden_size))
            if activation == 'relu':
                blocks.append(nn.ReLU())
            elif activation == 'sigmoid':
                blocks.append(nn.Sigmoid())
            elif activation == "leaky relu": 
                blocks.append(nn.LeakyReLU())
            else:
                raise ValueError('Not valid activation')
            input_size = hidden_size
            hidden_size = int(hidden_size * scale_factor)
        self.blocks = nn.Sequential(*blocks)
        self.linear = nn.Linear(input_size, target_size)
        
    def feature(self, x): 
        x = x.reshape(x.size(0), -1)
        x = self.blocks(x) 
        return x 
    
    def classify(self, x): 
        x = self.linear(x) 
        return x 
    
    def f(self, x): 
        x = self.feature(x)
        x = self.classify(x) 
        return x 
    
    def forward(self, x): 
        return self.f(x)


