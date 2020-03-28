import sys
sys.path.append('..')


import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from matplotlib import cm

from models.features import MobileNetV1_101_Features, MobileNetV1_Features_Org
from models.detection.ssd import create_mobilenetv1_ssd
from utils.blob import micronet_read_imgfile
from utils.models import load_checkpoint, model_report, model_save
from utils.models.visualize import SaveFeatures
from utils.misc import Timer
from utils import DEVICE, DATA_DIR, VISUALIZE_DIR

timer = Timer()
CHECKPOINT_NAME = 'mobilenetv1_101_features.pth'

# Input image preprocess
img_path = DATA_DIR + 'image/people.png'
draw_img, input_tensor = micronet_read_imgfile(img_path, height=529, width=961, batch_size=1)

"""
Model Prepare Block
"""
# Ours base_net
base_net = MobileNetV1_101_Features()
base_net = load_checkpoint(base_net, CHECKPOINT_NAME)

# Origin base_net
# base_net = MobileNetV1_Features_Org()

model = create_mobilenetv1_ssd(base_net.features, num_classes = 2)
model.eval()
model.to(DEVICE)
"""
End Model Prepare Block
"""

def model_inspection():
    model_report(model, input_shape=(3,529,961))
    print('SSD input shape: {}'.format(input_tensor.shape))
    print('Basenet output shape: {}'.format(model.base_net(input_tensor).shape))

    print('Output of extras layer size: ')
    x = model.base_net(input_tensor)
    for idx, layer in enumerate(model.extras):
        # print(layer)
        x = layer(x)
        print('Layer idx: {:2}, Size: {}'.format(idx, x.shape))
    print('\n')

    # Hook for filter inspection
    layer_num = 13
    hook_layer = list(model.base_net)[layer_num]
    activations = SaveFeatures(hook_layer)

    with torch.no_grad():
        output = model(input_tensor)
        print('Regression: {}, Classification: {}, Priors: {} boxes'
                .format(output[1].shape, output[0].shape, output[0].shape[1]))

    # Save every layer output
    filter_num = 0
    for filter_num in range(activations.features.shape[1]):
        mask = activations.features[0][filter_num].detach().cpu().numpy()
        mask *= 255.0/mask.max()
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.resize((1920, 1080), Image.BICUBIC).convert('RGB')
        image = draw_img.resize((1920, 1080), Image.BILINEAR)
        overlay_img = Image.blend(image, mask, 0.6)
        overlay_img.save(VISUALIZE_DIR + str(filter_num) + '.jpg')

def forward_test():
    with torch.no_grad():
        for idx in range(1):
            timer.start(key='Forward')
            output = model(input_tensor)
            timer.end(key='Forward')
            print('Regression shape: {}, Classification shape: {}'.format(output[1].shape, output[0].shape))

if __name__ == "__main__":
    # model_inspection()
    forward_test()
