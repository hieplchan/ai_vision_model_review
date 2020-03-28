import sys
sys.path.append('..')

import os
import logging
import itertools
import numpy as np
from PIL import Image
from matplotlib import cm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from models.features import MobileNetV1_Features_Org
from models.detection.ssd import create_mobilenetv1_ssd, MultiboxLoss
from models.detection.ssd import mobilenetv1_ssd_config as config
from utils.datasets.coco_dataset import CocoSubsetDetection, imgDirTrain, annFileTrain, imgDirVal, annFileVal
from utils.transforms.ssd_transforms import MatchPrior, TrainAugmentation, ValidateTransform
from utils.models import load_checkpoint, freeze_net_layers, save_training_params, model_save
from utils.misc import Timer
from utils import DEVICE, PYTORCH_MODELS_DIR

"""
Training parameters
"""
# Model Related Params
num_classes = 2
coco_subset_names = ['person']
checkpoint_folder = PYTORCH_MODELS_DIR + 'ssd/'
freeze_layer_type = 'freeze_base_net' # 'freeze_net', 'freeze_base_net', 'none_freeze'

# SGD Related Params
learning_rate = 1e-3 # Initial learning rate
base_net_lr = learning_rate # default: learning_rate
extra_layers_lr = learning_rate # default: learning_rate
momentum = 0.9 # Momentum value for optim
weight_decay = 5e-4 # Weight decay for SGD
gamma = 0.1 # Gamma update for SGD
scheduler = 'multi-step' # 'multi-step', 'cosine'
milestones = '80,100' # Milestones for MultiStepLR
t_max = 120 # T_max value for Cosine Annealing Scheduler
debug_steps = 50 # Debug log output frequency
min_loss = -10000.0
last_epoch = -1

# DataLoader Related Params
batch_size = 256
num_workers = 4
num_epochs = 30
validation_epochs = 1


def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    net.train(True)
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0

    for i, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if i and i % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            print('\nEpoch: {}, Step: {}, \nAverage Loss: {}, \nAverage Regression Loss {}, \nAverage Classification Loss: {}'
                    .format(epoch, i, avg_loss, avg_reg_loss, avg_clf_loss))
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0

def validate(loader, net, criterion, device):
    net.eval()
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    num = 0
    for _, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        num += 1

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
    return running_loss / num, running_regression_loss / num, running_classification_loss / num

timer = Timer()
if __name__ == "__main__":
    """
    Dataset Prepare Block
    """
    timer.start(key='Dataset Prepare')
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, config.iou_threshold)

    print('Training Dataset')
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std)
    train_dataset = CocoSubsetDetection(root=imgDirTrain, annFile=annFileTrain,
                                        catNames=coco_subset_names,
                                        transform=train_transform,
                                        target_transform=target_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=True)
    print('Number of training samples: {}'.format(train_dataset.__len__()))
    print('Number of batchs {} with batch size: {}\n'.format(len(train_dataloader), batch_size))

    print('Validate Dataset')
    val_transform = ValidateTransform(config.image_size, config.image_mean, config.image_std)
    val_dataset = CocoSubsetDetection(root=imgDirVal, annFile=annFileVal,
                                            catNames=coco_subset_names,
                                            transform=val_transform,
                                            target_transform=target_transform)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=True)
    print('Number of validating samples: {}'.format(val_dataset.__len__()))
    print('Number of batchs {} with batch size: {}\n'.format(len(val_dataloader), batch_size))
    timer.end(key='Dataset Prepare')
    """
    End Dataset Prepare Block
    """

    """
    Model Prepare Block
    """
    timer.start(key='Model Prepare')
    BASE_NET_CHECKPOINT_NAME = 'ssd_features_org.pth'
    EXTRAS_CHECKPOINT_NAME = 'ssd_extras_org.pth'
    FULL_SSD_CHECKPOINT_NAME = 'Epoch_29_Loss_1.0359992539181428.pth'
    base_net = MobileNetV1_Features_Org()
    model = create_mobilenetv1_ssd(base_net.features, num_classes = num_classes)
    # model.base_net = load_checkpoint(model.base_net, BASE_NET_CHECKPOINT_NAME)
    # model.extras = load_checkpoint(model.extras, EXTRAS_CHECKPOINT_NAME)
    model = load_checkpoint(model, checkpoint_name=FULL_SSD_CHECKPOINT_NAME, path=checkpoint_folder) #, path='/home/hiep/pytorch-vision/models/_models/ssd_train/'

    # Customize models
    if (freeze_layer_type ==  'freeze_net'):
        print('Freeze base net')
        freeze_net_layers(model.base_net)
        params = [
            {'params': model.extras.parameters(), 'lr': base_net_lr},
            {'params': itertools.chain(
                model.regression_headers.parameters(),
                model.classification_headers.parameters()
            )}]
    elif (freeze_layer_type ==  'freeze_base_net'):
        print('Freeze all the layers except prediction heads')
        freeze_net_layers(model.base_net)
        freeze_net_layers(model.extras)
        params = itertools.chain(model.regression_headers.parameters(),
                                model.classification_headers.parameters())
    elif (freeze_layer_type ==  'none_freeze'):
        print('None freeze')
        params = [
            {'params': net.base_net.parameters(), 'lr': base_net_lr},
            {'params': net.extras.parameters(), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}]
    else:
        print('Invalid freeze_layer_type')
    model.to(DEVICE)
    timer.end(key='Model Prepare')
    """
    End Model Prepare Block
    """

    """
    SGD Prepare Block
    """
    timer.start(key='SGD Prepare')
    criterion = MultiboxLoss(config.priors, iou_threshold=config.iou_threshold, neg_pos_ratio=config.neg_pos_ratio,
                             center_variance=config.center_variance, size_variance=config.size_variance, device=DEVICE)
    optimizer = torch.optim.SGD(params, lr=learning_rate, momentum=momentum,
                                weight_decay=weight_decay)
    print('Using SGD Optimizer with learning_rate={}, extra_layers_lr={}, base_net_lr={}, momentum={}, weight_decay={}.'
            .format(learning_rate, extra_layers_lr, base_net_lr, momentum, weight_decay))

    if scheduler == 'multi-step':
        milestones = [int(v.strip()) for v in milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones,
                                 gamma=gamma, last_epoch=last_epoch)
        print('Uses MultiStepLR scheduler with milestones={}, gamma={}, last_epoch={}.'
                .format(milestones, gamma, last_epoch))
    elif scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, t_max, last_epoch=last_epoch)
        print('Uses CosineAnnealingLR scheduler with t_max={}, last_epoch={}.'.format(t_max, last_epoch))
    else:
        print('Invalid scheduler {}.'.format(scheduler))
    timer.end(key='SGD Prepare')
    """
    End SGD Prepare Block
    """

    """
    Training Block
    """
    timer.start(key='Training')
    # print('Start training from epoch {}.'.format(last_epoch + 1))
    #
    # model_save(model, name='Epoch_{}_Loss_{}'.format(0, 0),
    #             input = torch.randn(1, 3, config.image_size[0], config.image_size[1]),
    #             path = checkpoint_folder)
    # checkpoint_path = checkpoint_folder + 'Epoch_{}_Loss_{}.pth'.format(0, 0)
    # save_training_params(0, model, optimizer, 0, checkpoint_path)

    print('### First Validation ###')
    timer.start(key='First Validate')
    val_loss, val_regression_loss, val_classification_loss = validate(val_dataloader, model, criterion, DEVICE)
    print('val_loss: {}'.format(val_loss))
    print('val_regression_loss: {}'.format(val_regression_loss))
    print('val_classification_loss: {}'.format(val_classification_loss))
    timer.end(key='First Validate')

    # for epoch in range(last_epoch + 1, num_epochs):
    #     timer.start(key='Training epoch: {}'.format(epoch))
    #     scheduler.step()
    #     train(train_dataloader, model, criterion, optimizer,
    #           device=DEVICE, debug_steps=debug_steps, epoch=epoch)
    #     if ((epoch % validation_epochs == 0) or (epoch == num_epochs - 1)):
    #         print('### Validation ###')
    #         val_loss, val_regression_loss, val_classification_loss = validate(val_dataloader, model, criterion, DEVICE)
    #         print('\nEpoch: {}'.format(epoch))
    #         print('val_loss: {}'.format(val_loss))
    #         print('val_regression_loss: {}'.format(val_regression_loss))
    #         print('val_classification_loss: {}'.format(val_classification_loss))
    #         model_save(model, name='Epoch_{}_Loss_{}'.format(epoch, val_loss),
    #                     input = torch.randn(1, 3, config.image_size[0], config.image_size[1]),
    #                     path = checkpoint_folder)
    #         checkpoint_path = checkpoint_folder + 'Epoch_{}_Loss_{}.pth'.format(epoch, val_loss)
    #         # save_training_params(epoch, model, optimizer, val_loss, checkpoint_path)
    #     timer.end(key='Training epoch: {}'.format(epoch))

    timer.end(key='Training')
    """
    End Training Block
    """
