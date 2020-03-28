import sys
sys.path.append('..')

import cv2
import time
import torch
import torch.nn as nn
import numpy as np

from models.features import MobileNetV1_101_Features, MobileNetV1_Features_Org
from utils.models import load_checkpoint, model_report
from utils.models.visualize import *
from utils.images import micronet_read_imgfile, PIL_to_cv2_image
from utils import DATA_DIR, VISUALIZE_DIR

# Input image preprocess
img_path = DATA_DIR + 'image/people.png'
img, input = micronet_read_imgfile(img_path)
img = PIL_to_cv2_image(img)

# CHECKPOINT_NAME = 'mobilenetv1_ssd_features_org.pth'
CHECKPOINT_NAME = 'mobilenetv1_101_features.pth'

if __name__ == "__main__":
    # model = MobileNetV1_Features_Org()
    model = MobileNetV1_101_Features()
    model = load_checkpoint(model, CHECKPOINT_NAME)
    model = model.eval()
    model_report(model, input_shape=(3,529,961))

    # Hook for inpection layer
    layer_num = 13
    hook_layer = list(model.features)[layer_num]
    print(hook_layer)
    activations = SaveFeatures(hook_layer)

    with torch.no_grad():
        for idx in range(1):
            start_time_mark = time.time()

            output = model(input)
            print('Input shape: {}'.format(input.shape))
            print('Input memory size: {:06} bytes'.format(input.element_size() * input.nelement()))
            print('Layer output shape: {}'.format(activations.features.shape))
            print('Layer output memory size: {:06} bytes'.format(activations.features.element_size() * activations.features.nelement()))

            # Filter output draw
            filter_num = 0
            for filter_num in range(activations.features.shape[1]):
                mask = activations.features[0][filter_num].detach().cpu().numpy()
                mask *= 255.0/mask.max()
                mask = mask.astype(np.uint8)
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                overlay_image = cv2.addWeighted(img, 0.3, mask, 1.0, 0)
                cv2.imwrite(VISUALIZE_DIR + str(filter_num) + '.jpg', overlay_image)

            end_time_mark = time.time()
            print('Total time: {:06.3f} ms'.format((end_time_mark - start_time_mark)*1000))
