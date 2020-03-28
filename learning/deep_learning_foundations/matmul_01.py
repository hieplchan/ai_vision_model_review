import sys
sys.path.append('../..')

from pathlib import Path
from fastai import datasets
import pickle, gzip, math, torch, matplotlib as mpl
import matplotlib.pyplot as plt
from torch import tensor
import numpy

from utils import IMAGE_DIR, DEVICE
from utils.misc import timer

"""
DATASET PREPARE BLOCK
"""

MNIST_URL='http://deeplearning.net/data/mnist/mnist.pkl'

path = datasets.download_data(MNIST_URL, ext='.gz')
# print('mnist path: {}: '.format(path))

with gzip.open(path, 'rb') as f:
    ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')

x_train, y_train, x_valid, y_valid = map(tensor, (x_train, y_train, x_valid, y_valid))
row, col = x_train.shape
# print('mnist dataset review')
# print(x_train, x_train.shape, y_train, y_train.shape, y_train.min(), y_train.max())
# print(row, col) # c = 28*28 pixel

mpl.rcParams['image.cmap'] = 'gray'
img = x_train[0]
# print(img.view(28,28).type())
plt.imsave(IMAGE_DIR + 'mnist_review.jpg', img.view((28,28)))

"""
END DATASET PREPARE BLOCK
"""

"""
SIMPLE LINEAR MODEL BLOCK
"""

# input: 28*28, output: 0 to 9
weights = torch.randn(784,10)
bias = torch.zeros(10)

"""
END SIMPLE LINEAR MODEL BLOCK
"""

"""
OPS BLOCK
    @visualize matmul: http://matrixmultiplication.xyz/
    @elementwise explained: https://www.youtube.com/watch?v=2GPZlRVhQWY
    @Frobenius norm: https://machinelearningcoban.com/2017/10/20/fundaml_matrices/
    @broadcasting: https://vimentor.com/vi/lesson/29-ma-tran-va-cac-phep-toan-dai-so-khi-lam-viec-voi-numpy-array
    @broadcasting visualize: https://docs.google.com/spreadsheets/d/1bIPBcf-p9iqNG8BGmIVlJCFa4jEsbOZvcPXGTYe5pjI/edit
    @einstein summation: https://obilaniu6266h16.wordpress.com/2016/02/04/einstein-summation-in-numpy/
                        https://medium.com/datadriveninvestor/how-einstein-summation-works-and-its-applications-in-deep-learning-1649f925aaae
"""

# test variable
m1 = x_valid[:5]
m2 = weights
a = tensor([10., 6, -4])
b = tensor([2., 8, 7])
c = tensor([10.,20,30])
m = tensor([[1., 2, 3], [4,5,6], [7,8,9]]) # rank 2 matrix
# print(m1.shape, m2.shape)
# print(a, b)
# print(m)

# test matmul_pure_python
def matmul_pure_python(a, b):
    a_row, a_col = a.shape
    b_row, b_col = b.shape
    # check valid dimension of matrix
    assert a_col == b_row
    c = torch.zeros(a_row, b_col)
    for i in range(a_row):
        for j in range(b_col):
            for k in range(a_col): # or b_row
                c[i,j] += a[i,k] * b[k,j]
    return c

for i in range(0):
    timer.start(key='matmul_element_wise measure')
    t1 = matmul_element_wise(m1, m2) # significant speed up from 383ms downto 0.735ms
    timer.end(key='matmul_element_wise measure')
    print(t1.shape)

# test matmul_element_wise
# print(a + b)
# print((a < b).float().mean()) # which percent of element in a less than b
# print((m*m).sum().sqrt()) # Frobenius norm

def matmul_element_wise(a, b):
    a_row, a_col = a.shape
    b_row, b_col = b.shape
    # check valid dimension of matrix
    assert a_col == b_row
    c = torch.zeros(a_row, b_col)
    for i in range(a_row):
        for j in range(b_col):
            # entire a row & antire b columm
            c[i,j] += (a[i,:] * b[:,j]).sum()
    return c

for i in range(0):
    timer.start(key='matmul_element_wise measure')
    t1 = matmul_element_wise(m1, m2) # significant speed up from 383ms downto 0.735ms
    timer.end(key='matmul_element_wise measure')
    print(t1.shape)

# Compare result of matmul_pure_python and matmul_element_wise
# print(torch.allclose(matmul_pure_python(m1, m2), matmul_element_wise(m1, m2), rtol=1e-3, atol=1e-5))

# test matmul_element_wise_broadcasting
# print(a > 0) # broadcast scalar 0 to [0, 0, 0]
# print(2*m)
#
# print(c.shape, m.shape)
# print(c + m) # broadcasting c to m shape
# t = c.expand_as(m) # broadcast among row
# print(t, t.storage()) # real c form when broadcasting: 3*3 but memory size is still not change (3*1) - not memory intensive
# print(t.stride(), t.shape) # t.stride (0, 1): from row to row take 0 step, from col to col take 1 step
# # You can index with the special value [None] or use unsqueeze() to convert a 1-dimensional array into a 2-dimensional array (although one of those dimensions has value 1).
# print(c, c.unsqueeze(0), c.unsqueeze(1))
# print(c.shape, c.unsqueeze(0).shape, c.unsqueeze(1).shape)
# print(c.shape, c[None].shape, c[:,None].shape) # should use None instead of unsqueeze, same result of above
# print(c.shape, c[None].shape, c[...,None].shape) # same as above but not care about shape
# print(c[:,None].expand_as(m)) # broadcast among columm instead of row
# print(c[:,None] + m)

def matmul_element_wise_broadcasting(a, b):
    a_row, a_col = a.shape
    b_row, b_col = b.shape
    # check valid dimension of matrix
    assert a_col == b_row
    c = torch.zeros(a_row, b_col)
    for i in range(a_row):
            # entire row of c[i]
            c[i,:] += (a[i].unsqueeze(-1) * b).sum(dim=0) # -1 means last dimension, sum over row
            # c[i,:] += (a[i,:][...,None] * b).sum(dim=0) # same as above
    return c

# print(m1.shape, m2.shape)
# print((m1[0].unsqueeze(-1) * m2).shape)
# print((m1[0].unsqueeze(-1) * m2).sum(dim=0).shape)

# for i in range(0):
#     timer.start(key='matmul_element_wise_broadcasting measure')
#     t1 = matmul_element_wise_broadcasting(m1, m2) # significant speed up from 383ms downto 0.735ms downto 0.229ms
#     timer.end(key='matmul_element_wise_broadcasting measure')
# print(t1.shape)

# Compare result of matmul_pure_python and matmul_element_wise_broadcasting
# print(torch.allclose(matmul_pure_python(m1, m2), matmul_element_wise_broadcasting(m1, m2), rtol=1e-3, atol=1e-5))

# broadcasting rules
# print(c[None,:], c[None,:].shape, c[:,None].shape)
# print(c[None,:]*c[:,None]) # (1,3) and (3,1): 1 dimension is broadcast to 3

# test matmul_einstein_summation
def matmul_einstein_summation(a, b):
    return torch.einsum('ik,kj->ij', a, b)
    # return numpy.einsum('ik,kj->ij', a, b)

for i in range(0):
    timer.start(key='matmul_einstein_summation measure')
    t1 = matmul_einstein_summation(m1, m2) # significant speed up from 383ms downto 0.735ms downto 0.229ms down to 0.023ms
    timer.end(key='matmul_einstein_summation measure')
    print(t1.shape)

# Compare result of matmul_pure_python and matmul_element_wise_broadcasting
# print(torch.allclose(matmul_pure_python(m1, m2), matmul_einstein_summation(m1, m2), rtol=1e-3, atol=1e-5))


# test pytorch ops
# this to fast because it splits matrix to smaller one to fit into CPU catch, not RAM access by using BLAS (cuBLAS, MKL)
def matmul_pytorch(a, b):
    return a.matmul(b)
    # return t1 = m1@m2 # same as above

for i in range(0):
    timer.start(key='matmul_einstein_summation measure')
    t1 =  matmul_pytorch(m1, m2)# significant speed up from 383ms downto 0.735ms downto 0.229ms down to 0.023ms downto 0.005ms
    timer.end(key='matmul_einstein_summation measure')

"""
END OPS BLOCK
"""
