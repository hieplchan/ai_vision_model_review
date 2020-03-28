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

MNIST_URL='http://deeplearning.net/data/mnist/mnist.pkl'

path = datasets.download_data(MNIST_URL, ext='.gz')
print('mnist path: {}: '.format(path))

def get_data():
    '''
    Split MNIST dataset to train and validation set
    '''
    with gzip.open(path, 'rb') as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding='latin-1')

    return map(tensor, (x_train, y_train, x_valid, y_valid))

def normalize(x, mean, standard):
    '''
    Normalize using broadcasting
    '''
    return (x - mean)/standard
