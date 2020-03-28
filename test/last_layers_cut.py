import sys
sys.path.append('..')

import cv2
import time
import torch
import torch.nn as nn
import numpy as np

from models import *
from utils import *

# Input image preprocess
img_path = DATA_PATH + 'image/people.png'
img = cv2.imread(img_path)
input = image_preprocess_mobilenetv1(img)

CHECKPOINT_NAME = 'mobilenetv1_101.pth'

if __name__ == "__main__":
    model = MobileNetV1_101()

    # Load model checkpoint
    model = load_checkpoint(model, CHECKPOINT_NAME)

    # Depend on your model architecture
    # print(list(model.children())[:-4][0].state_dict())

    # Cut last layer of model for another purpose
    new_model = MobileNetV1_101_Features()
    new_model.features.load_state_dict(list(model.children())[:-4][0].state_dict())

    # Save model to disk .pth and .onnx
    model_save(new_model, name = 'mobilenetv1_101_features')

    # Inspect model detail
    model_report(new_model)

    with torch.no_grad():
        for idx in range(10):
            start_time_mark = time.time()

            output = new_model(input)
            print(output.shape)

            end_time_mark = time.time()
            print('Total time: {:06.3f} ms'.format((end_time_mark - start_time_mark)*1000))
