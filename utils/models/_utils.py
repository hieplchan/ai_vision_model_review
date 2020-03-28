import os
import cv2
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchsummary import summary

from utils import DEVICE, PYTORCH_MODELS_DIR
from .flops_counter import get_model_complexity_info

def model_save(model, input_shape, batch_size=1, output_name='pytorch-vision-models', path='./', onnx_save=False):
    """
    Save Pytorch model to .pth and .onnx format.

    Arguments:
        model (nn.Module): model to save
        input_shape (chanel, height, width): image input shape of model
        batch_size (int): model batch size, default is 1
        output_name (string): name of the output file
        path (string): location to save file
        onnx_save (bool): set True to save .onnx file
    """
    input = torch.randn(batch_size, input_shape[0], input_shape[1], input_shape[2])
    _model = model.to(DEVICE)

    # Save pytorch model
    torch.save(_model.state_dict(), path + output_name + '.pth')

    if onnx_save==True:
        # Convert pytorch models to ONNX and save to file
        torch_output = torch.onnx._export(_model, input.to(DEVICE), path + output_name + '.onnx', export_params=True)

def model_report(model, input_shape, detail=True):
    """
    Model Report for easy inspection

    Arguments:
        model (nn.Module): model to inspection
        input_shape (chanel, height, width): image input shape of model
        detail (bool): set True for detail inspection
    """
    _model = model.to(DEVICE)

    print('========================= MODEL SUMMARY =========================')
    summary(_model, input_shape)
    print('\n\n')

    if detail==True:
        print('========================= MODEL DETAIL REPORT ==========================')
        flops, params = get_model_complexity_info(_model, input_shape, as_strings=True, print_per_layer_stat=True)
        print('Flops:  ' + flops)
        print('Params: ' + params)

def load_checkpoint(model, checkpoint_name, path=PYTORCH_MODELS_DIR):
    """
    Load Pytorch .pth model file

    Arguments:
        model (nn.Module): model to load checkpoint
        checkpoint_name (string): name of the checkpoint file
        path (string): location to checkpoint file, default is 'models/_models/pytorch'

    Returns:
        model (nn.Module): model with loaded params and put to DEVICE
    """
    model_checkpoint = torch.load(path + checkpoint_name, map_location = DEVICE)
    model.load_state_dict(model_checkpoint)
    return model.to(DEVICE)

def freeze_net_layers(net):
    """
    Freeze part of model that not use gradients

    Arguments:
        net: part of model to freeze, example: ssd.base_net
    """
    for param in model.parameters():
        param.requires_grad = False

"""
Developing Functions Block
"""

def load_checkpoint_tar(model, checkpoint_name, path=PYTORCH_MODELS_DIR):
    """
    TAR file checkpoint of origin gitlab (MobileNetV3)
    """
    model_checkpoint = torch.load(path + checkpoint_name, map_location = DEVICE)
    model.load_state_dict({k.replace('module.', ''): v for k, v in model_checkpoint['state_dict'].items()})
    return model.to(DEVICE)

def save_training_params(epoch, net_state_dict, optimizer_state_dict, best_score, checkpoint_path):
    torch.save({
        'epoch': epoch,
        'state_dict': net_state_dict,
        'optimizer': optimizer_state_dict,
        'best_score': best_score
    }, checkpoint_path)

"""
End Developing Functions Block
"""
