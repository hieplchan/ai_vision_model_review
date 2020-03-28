import sys
sys.path.append('..')

import os
import torch
import models

ROOT_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../'
MODELS_DIR = ROOT_DIR + 'models/'
PYTORCH_MODELS_DIR = MODELS_DIR + '_models/pytorch/'
OPENVINO_MODELS_DIR = MODELS_DIR + '_models/openvino/'
LABEL_PATH = MODELS_DIR + 'labels/'
DATA_DIR = ROOT_DIR + 'data/'
IMAGE_DIR = DATA_DIR + 'image/'
VISUALIZE_DIR = DATA_DIR + 'visualize/filter/'

print('\n')
print('========================== Pytorch Info =========================')
if not torch.cuda.is_available():
    DEVICE = 'cpu'
    print('Working on CPU')
else:
    DEVICE = 'cuda'
    print('Working on GPU')
    print('CUDA devices count: {}'.format(torch.cuda.device_count()))
    print('CUDA devices name: {}'.format(torch.cuda.get_device_name(0)))
print('=================================================================')
print('\n')
