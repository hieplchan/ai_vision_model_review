import sys
sys.path.append('../..')

import math
import torch
from torch.nn import init

from matmul_01 import matmul_pytorch as matmul
from mnist_dataset import get_data, normalize
from utils.misc import timer, test_near_zero, test_near

"""
DATASET BLOCK
"""
x_train, y_train, x_valid, y_valid = get_data()
x_train_mean, x_train_standard = x_train.mean(), x_train.std()
print('x_train_mean: {0:0.6f}, x_train_standard: {0:0.6f}'.format(x_train_mean, x_train_standard))

# We want mean and standard deviation to be: 0 and 1
x_train = normalize(x_train, x_train_mean, x_train_standard)
# Use training (not validate) mean and standard for validation set, keep them at the same scale
x_valid = normalize(x_valid, x_train_mean, x_train_standard)
x_train_mean, x_train_standard = x_train.mean(), x_train.std()
print('Normalized x_train_mean: {0:0.6f}, Normalized x_train_standard: {0:0.6f}'.format(x_train_mean, x_train_standard))

# Check if mean and standard close to 0 and 1 ???
test_near_zero(x_train_mean)
test_near_zero(1 - x_train_standard)

y_train, y_valid = y_train.float(), y_valid.float()
"""
END DATASET BLOCK
"""

"""
MODEL BLOCK
    @basic architecture:    1 hidden layers
                            1 output activations (mean square error)
    @kaiming / he initialization:   random without kaimning/he initialization is very bad: mean~2.7, std~26 (compare to 0.0117 and 1.0037)
                                    relu init mean and std decrease 2 times each layers (Vanishing Gradient Problem)
"""
num_sample, num_image_size = x_train.shape # shape of train set
num_class = y_train.max() + 1 # number of class
print('num_sample: {}, num_image_size: {}, num_class: {}'.format(num_sample, num_image_size, num_class))

num_hidden = 50 # number of hidden layers

# pytorch weights initialization
# weight_1 = torch.zeros(num_image_size, num_hidden)
# init.kaiming_normal_(weight_1, mode='fan_out') # pytorch weight initialization, something badly with pytorch code

# simplified kaiming / he weights initialization (standard * 2)
weight_1 = torch.randn(num_image_size, num_hidden)*math.sqrt(2/num_image_size)
bias_1 = torch.zeros(num_hidden)
weight_2 = torch.randn(num_hidden, 1)/math.sqrt(num_hidden)
bias_2 = torch.zeros(1)
print('weight_1 mean: {} and standard {}'.format(weight_1.mean(), weight_1.std()))


def linear_layer(x, weight, bias):
    return x@weight + bias

def relu(x):
    '''
    Replace negative with zero
    '''
    # return x.clamp_min(0.)
    return x.clamp_min(0.) - 0.5 # is this new activations helpful?

layer_1 = relu(linear_layer(x_valid, weight_1, bias_1))
print('layer_1 mean: {} and standard {}'.format(layer_1.mean(), layer_1.std()))

def model(x):
    layer_1 = linear_layer(x, weight_1, bias_1)
    layer_2 = relu(layer_1)
    layer_3 = linear_layer(layer_2, weight_2, bias_2)
    return layer_3

# forward pass time test
for i in range(0):
    timer.start(key='forward pass measure')
    _ =  model(x_valid)# significant speed up from 383ms downto 0.735ms downto 0.229ms down to 0.023ms downto 0.005ms
    timer.end(key='forward pass measure')
assert model(x_valid).shape == torch.Size([x_valid.shape[0],1])
print('model output shape: {}'.format(model(x_valid).shape)) #[10000, 1]
"""
END MODEL BLOCK
"""

"""
LOSS FUNCTION BLOCK
    @simplified loss: MSE - mean squared error
"""
def mse_loss(predicted, observed):
    return (observed - predicted.squeeze(-1)).pow(2).mean()

preds = model(x_train)
print('predicts shape: {}'.format(preds.shape))

print('mse loss calculate on forward pass: {}'.format(mse_loss(preds, y_train)))

"""
END LOSS FUNCTION BLOCK
"""

"""
GRADIENT & BACKWARD PASS BLOCK
    @chain rules: https://www.youtube.com/watch?v=fDeAJspBEnM
"""
def mse_loss_grad(input, observed):
    # gradient of loss with respect to output of previous layer
    input.g = 2.0*(input.squeeze(-1) - observed).unsqueeze(-1) / input.shape[0]
def relu_grad(input, output):
    # grad of relu with respect to input activations
    input.g = (input>0).float() * output.g
def linear_grad(input, output, weight, bias):
    input.g = output.g @ weight.t()
    weight.g = (input.unsqueeze(-1) * output.g.unsqueeze(1)).sum(0)
    bias.g = output.g.sum(0)

def forward_and_backward(input, observed):
    # forward pass
    l1 = input@weight_1 + bias_1
    l2 = relu(l1)
    output = l2@weight_2 + bias_2

    # we don't actually need the loss in backward!
    loss = mse_loss(output, observed)

    # backward pass
    mse_loss_grad(output, observed)
    linear_grad(l2, output, weight_2, bias_2)
    relu_grad(l1, l2)
    linear_grad(input, l1, weight_1, bias_1)

forward_and_backward(x_train, y_train)

# Save for testing against later
w1g = weight_1.g.clone()
w2g = weight_2.g.clone()
b1g = bias_1.g.clone()
b2g = bias_2.g.clone()
ig  = x_train.g.clone()

# Using Pytorch to check result
# requires_grad_: True keep track of forward functions to do backward pass
xt2 = x_train.clone().requires_grad_(True)
w12 = weight_1.clone().requires_grad_(True)
w22 = weight_2.clone().requires_grad_(True)
b12 = bias_1.clone().requires_grad_(True)
b22 = bias_2.clone().requires_grad_(True)

def forward(inp, targ):
    # forward pass:
    l1 = inp @ w12 + b12
    l2 = relu(l1)
    out = l2 @ w22 + b22
    # we don't actually need the loss in backward!
    return mse_loss(out, targ)

loss = forward(xt2, y_train)
loss.backward()
test_near(w22.grad, w2g)
test_near(b22.grad, b2g)
test_near(w12.grad, w1g)
test_near(b12.grad, b1g)
test_near(xt2.grad, ig)

"""
END GRADIENT & BACKWARD PASS BLOCK
"""
