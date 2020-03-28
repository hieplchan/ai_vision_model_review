import sys
sys.path.append('../..')

import math
import torch
from torch import nn

from mnist_dataset import get_data, normalize
from utils.misc import timer, test_near_zero


"""
DATASET BLOCK
"""
x_train, y_train, x_valid, y_valid = get_data()

x_train_mean, x_train_standard = x_train.mean(), x_train.std()
x_train = normalize(x_train, x_train_mean, x_train_standard)
x_valid = normalize(x_valid, x_train_mean, x_train_standard)
x_train_mean, x_train_standard = x_train.mean(), x_train.std()
test_near_zero(x_train_mean)
test_near_zero(1 - x_train_standard)

y_train, y_valid = y_train.float(), y_valid.float()
"""
END DATASET BLOCK
"""

"""
LAYER AS CLASS BLOCK
"""
# class Relu():
#     def __call__(self, inp):
#         self.inp = inp
#         self.out = inp.clamp_min(0.) - 0.5
#         return self.out

#     def backward(self):
#         self.inp.g = (self.inp>0).float()*self.out.g

# class Lin():
#     def __init__(self, w, b):
#         self.w = w
#         self.b = b

#     def __call__(self, inp):
#         self.inp = inp
#         self.out = inp@self.w + self.b
#         return self.out

#     def backward(self):
#         self.inp.g = self.out.g@self.w.t()
#         # Creating a giant outer product, just to sum it, is inefficient!
#         self.w.g = (self.inp.unsqueeze(-1)@self.out.g.unsqueeze(1)).sum(0)
#         self.b.g = self.out.g.sum(0)

# class MSE():
#     def __call__(self, inp, targ):
#         self.inp = inp
#         self.targ = targ
#         self.out = (self.inp.squeeze(-1) - self.targ).pow(2).mean()
#         return self.out

#     def backward(self):
#         self.inp.g = 2.0*(self.inp.squeeze(-1) - self.targ).unsqueeze(-1)/self.targ.shape[0]
"""
END LAYER AS CLASS BLOCK
"""


"""
MODULE BLOCK
"""
class Module():
    def __call__(self, *args):
        self.args = args
        self.out = self.forward(*args)
        return self.out

    def forward(self):
        raise Exception('Not Implemented')

    def backward(self):
        self.bwd(self.out, *self.args)

class Relu(Module):
    def forward(self, inp):
        return inp.clamp_min(0.) - 0.5

    def bwd(self, out, inp):
        inp.g = (inp>0).float()*out.g

class Lin(Module):
    def __init__(self, w, b):
        self.w, self.b = w, b

    def forward(self, inp):
        return inp@self.w + self.b

    def bwd(self, out, inp):
        inp.g = out.g@self.w.t()
        # self.w.g = inp.t()@out.g # transpose
        self.w.g = torch.einsum("bi,bj->ij", inp, out.g) # einstein summation
        # self.w.g = (inp.unsqueeze(-1)@out.g.unsqueeze(1)).sum(0) # inefficient code
        self.b.g = out.g.sum(0)

class MSE(Module):
    def forward(self, inp, targ):
        return (inp.squeeze(-1) - targ).pow(2).mean()

    def bwd(self, out, inp, targ):
        inp.g = 2.0*(inp.squeeze(-1) - targ).unsqueeze(-1)/targ.shape[0]
"""
END MODULE BLOCK
"""

"""
MODEL BLOCK
"""
class Model():
    def __init__(self, w1, b1, w2, b2):
        self.layers = [Lin(w1,b1), Relu(), Lin(w2,b2)]
        self.loss = MSE()

    def __call__(self, x, targ):
        for layer in self.layers:
            x = layer(x)
        return self.loss(x, targ)

    def backward(self):
        self.loss.backward()
        for layer in reversed(self.layers):
            layer.backward()

num_sample, num_image_size = x_train.shape
num_class = y_train.max() + 1
num_hidden = 50

# Kaiming He weights initialization
w1 = torch.randn(num_image_size, num_hidden)*math.sqrt(2/num_image_size)
b1 = torch.zeros(num_hidden)
w2 = torch.randn(num_hidden, 1)/math.sqrt(num_hidden)
b2 = torch.zeros(1)

w1.g, b1.g, w2.g, b2.g = [None]*4
model = Model(w1, b1, w2, b2)

timer.start(key='forward')
for i in range(1):
    loss = model(x_train, y_train)
timer.end(key='forward')

timer.start(key='backward')
for i in range(1):
    model.backward()
timer.end(key='backward')

"""
END MODEL BLOCK
"""

"""
PYTORCH MODEL BLOCK
"""
class Model(nn.Module):
    """
    Pytorch Module for speed compare
    """
    def __init__(self, num_image_size, num_hidden, num_out):
        super().__init__()
        self.layers = [nn.Linear(num_image_size, num_hidden), nn.ReLU(), nn.Linear(num_hidden, num_out)]
        self.loss = MSE()

    def __call__(self, x, targ):
        for layer in self.layers:
            x = layer(x)
        return self.loss(x.squeeze(), targ)

model = Model(num_image_size, num_hidden, 1)

timer.start(key='forward')
for i in range(1):
    loss = model(x_train, y_train)
timer.end(key='forward')

timer.start(key='backward')
for i in range(1):
    loss.backward()
timer.end(key='backward')

"""
END PYTORCH MODEL BLOCK
"""
