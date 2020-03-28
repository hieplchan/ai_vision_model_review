import sys
sys.path.append('..')

import cv2
import time
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models

from models import *
from utils import *

VISUALIZE_DIR = '../features_visualize/'

if __name__ == "__main__":
    model = models.vgg16(pretrained=True)

    # model_report(model)
    # print(len(list(model.features)))
    # if (type(list(model.features)[7]) == torch.nn.modules.conv.Conv2d):
    #     print('convolution layer')
    # print(list(model.features)[7].weight.shape[0]) # [number_of_filter, input_channels, h, w].

    m = model.eval()
    fv = FilterVisualizer(m)

    start_time_mark = time.time()

    layer_num = 0
    filter_num = 0
    for layer_num in range(len(list(model.features))):
        if (type(list(model.features)[layer_num]) == torch.nn.modules.conv.Conv2d):
            print('Convolution layer: ' + str(layer_num))
            for filter_num in range(list(model.features)[layer_num].weight.shape[0]):
                img = fv.visualize(121, m.features[layer_num], filter=filter_num, upscaling_steps=4, upscaling_factor=1.5, opt_steps=20, blur=3, print_losses=False)
                print('Done layer ' + str(layer_num) + ' filter ' + str(filter_num))
                plt.figure(figsize=(7,7))
                plt.imsave(VISUALIZE_DIR + 'layer' + str(layer_num) + '_filter' + str(filter_num) + '.jpg',img)
                plt.clf()

    end_time_mark = time.time()
    print('Total time: {:06.3f} ms'.format((end_time_mark - start_time_mark)*1000))
