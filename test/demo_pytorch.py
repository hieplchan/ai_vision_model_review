import sys
sys.path.append('..')

import cv2
import time
import torch
import numpy as np

from models import *
from utils import *

import torchvision.models as models

img_path = DATA_PATH + 'image/people.png'
img = cv2.imread(img_path)
input = image_preprocess_imagenet(img)

CHECKPOINT_NAME = 'mobilenetv2.pth'

if __name__ == "__main__":
    # Define model architecture
    # TODO: list of architectures
    # model = MobileNetV2()
    model = models.densenet161(pretrained=True)
    model.to(DEVICE)

    # Load model checkpoint
    # model = load_checkpoint(model, CHECKPOINT_NAME)

    # Save model to disk .pth and .onnx
    # model_save(model, name = 'densenet161')

    # Inspect model detail
    # model_report(model)

    with torch.no_grad():
        for idx in range(10000):
            start_time_mark = time.time()

            output = model(input)
            print(output.shape)

            end_time_mark = time.time()

            # _, predicted = torch.max(output, 1)
            print('Total time: {:06.3f} ms'.format((end_time_mark - start_time_mark)*1000))
