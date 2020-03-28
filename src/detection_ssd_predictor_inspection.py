import sys
sys.path.append('..')
sys.path.append('/home/hiep/pytorch-ssd')

import cv2
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from matplotlib import cm

from models.features import MobileNetV1_Features_Org
from models.detection.ssd import create_mobilenetv1_ssd, create_mobilenetv1_ssd_predictor
from utils.blob import micronet_read_imgfile
from utils.models import load_checkpoint, model_report, model_save
from utils.models.visualize import SaveFeatures
from utils.misc import Timer
from utils import DEVICE, DATA_DIR, PYTORCH_MODELS_DIR, LABEL_PATH

timer = Timer()

# Input image preprocess
img_path = DATA_DIR + 'image/people.png'
out_img_path = DATA_DIR + 'image/output.png'
draw_img, input_tensor = micronet_read_imgfile(img_path, height=300, width=300, batch_size=1)
orig_image = cv2.imread(img_path)
image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)

"""
Model Prepare Block
"""
# Ours base_net
checkpoint_folder = PYTORCH_MODELS_DIR + 'ssd/'
FULL_SSD_CHECKPOINT = 'Epoch_29_Loss_1.0359992539181428.pth'
ORIGIN_SSD_CHECKPOINT =  'mobilenet-v1-ssd-mp-0_675.pth'
label_path = LABEL_PATH + 'voc-model-labels.txt'
class_names = [name.strip() for name in open(label_path).readlines()]

base_net = MobileNetV1_Features_Org()
model = create_mobilenetv1_ssd(base_net.features, num_classes = len(class_names))
model = load_checkpoint(model, checkpoint_name=ORIGIN_SSD_CHECKPOINT, path=checkpoint_folder) #, path='/home/hiep/pytorch-vision/models/_models/ssd_train/'
predictor = create_mobilenetv1_ssd_predictor(model, candidate_size=200)
model.to(DEVICE)
"""
End Model Prepare Block
"""

def forward_test():
    with torch.no_grad():
        for idx in range(1):
            timer.start(key='Forward')
            output = model(input_tensor)
            timer.end(key='Forward')
            print('Regression shape: {}, Classification shape: {}'.format(output[1].shape, output[0].shape))

def predictor_head_inspection():
    boxes, labels, probs = predictor.predict(image, 10, 0.4)
    print('probs len: {}, max: {}, min: {}'.format(len(probs), max(probs), min(probs)))
    print('boxes shape: {}'.format(boxes.shape))
    print('labels shape: {}'.format(labels.shape))

    for i in range(boxes.size(0)):
        box = boxes[i, :]
        label = f"{class_names[labels[i]]}: {probs[i]:.2f}"
        print(label)
        if (label == 'person'):
            cv2.rectangle(orig_image, (box[0], box[1]), (box[2], box[3]), (255, 255, 0), 4)
            cv2.putText(orig_image, label,
                        (box[0] + 20, box[1] + 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,  # font scale
                        (255, 0, 255),
                        2)  # line type
    cv2.imwrite(out_img_path, orig_image)
    print(f"Found {len(probs)} objects. The output image is {out_img_path}")

if __name__ == "__main__":
    # forward_test()
    predictor_head_inspection()
