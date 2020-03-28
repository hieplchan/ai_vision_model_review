import sys
sys.path.append('..')

import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

from utils import DEVICE

def imagenet_read_imgfile(img, batch_num = 1):
    """
    Transform image for imagenet validate
    """
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # Use this for nasnetamobile
    # normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
    #                                  std=[0.5, 0.5, 0.5])

    transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ])

    batch_img = []
    for i in range(batch_num):
        batch_img.append(transform(img))
    input_tensor = torch.stack(batch_img, 0)

    return input_tensor.to(DEVICE)

def micronet_read_imgfile(path, height=529, width=961, batch_size=1):
    img = Image.open(path).convert('RGB')
    transform = micronet_image_transform(width, height)
    batch_img = []
    for i in range(batch_size):
        batch_img.append(transform(img))
    img_tensor = torch.stack(batch_img, 0)
    return img, img_tensor.to(DEVICE)

def micronet_resolution_validate(height, width, scale_factor=1.0, output_stride=16):
    target_width = (int(width) // output_stride) * output_stride + 1
    target_height = (int(height) // output_stride) * output_stride + 1
    scale = np.array([height / target_height, width / target_width])
    print('Origin Width: {}, Height: {}'.format(width, height))
    print('Scale Factor: {}, Output Stride: {}'.format(scale_factor, output_stride))
    print('Targer Width: {}, Height: {}'.format(target_width, target_height))
    print('\n')
    return target_width, target_height, scale

def micronet_image_transform(height, width):
    target_width, target_height, scale = micronet_resolution_validate(height, width)

    # Normalize image = (image - mean) / std
    r_mean, g_mean, b_mean = (0.5,  0.5, 0.5)
    r_std, g_std, b_std = (0.5, 0.5, 0.5)
    normalize = transforms.Normalize(mean=(r_mean, g_mean, b_mean),
                                     std=(r_std, g_std, b_std))
    transform = transforms.Compose([
                transforms.Resize((target_height, target_width), interpolation= Image.BICUBIC), #Image.LANCZOS, Image.NEAREST, Image.BICUBIC, Image.BILINEAR
                transforms.ToTensor(),
                normalize])

    return transform

def PIL_to_cv2_image(PIL_Image):
    cv2_img = np.array(PIL_Image)
    # Convert RGB to BGR
    cv2_img = cv2_img[:, :, ::-1].copy()
    return cv2_img
