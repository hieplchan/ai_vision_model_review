import sys
sys.path.append('..')

import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets


"""
Pytorch model params
"""

from models import *
from utils import *

CHECKPOINT_NAME = 'resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar'

"""
Validate config params
"""
MODEL_DIR = ''
IMAGENET_DIR = '/media/WINDOW/Users/ari/Desktop/ImageNetVal'
BATCH_SIZE = 400
NUM_WORKERS = 6

# Imagenet Validate Set
# Why choose normalize and crop params like that???
r_mean, g_mean, b_mean = (0.485,  0.456, 0.406)
r_std, g_std, b_std = (0.229, 0.224, 0.225)
if (CHECKPOINT_NAME == 'nasnetamobile.pth'):
    r_mean, g_mean, b_mean = (0.5,  0.5, 0.5)
    r_std, g_std, b_std = (0.5, 0.5, 0.5)

normalize = torchvision.transforms.Normalize(mean=(r_mean, g_mean, b_mean),
                                 std=(r_std, g_std, b_std))

dataloader = torch.utils.data.DataLoader(
    datasets.ImageFolder(IMAGENET_DIR, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True)

if __name__ == "__main__":
    model = torchvision.models.densenet161(pretrained=True)
    model.eval()
    # model = load_checkpoint_tar(model, CHECKPOINT_NAME)

    if (torch.cuda.device_count() > 0):
        model.cuda()
        cudnn.benchmark = True
        cudnn.deterministic = True
        if (torch.cuda.device_count() > 1):
            model = nn.DataParallel(model, device_ids = 0).cuda()

    # eval
    loss = 0
    top1 = 0
    top5 = 0
    criterion = nn.CrossEntropyLoss()
    timer = time.time()
    for batch_idx, (data, target) in enumerate(dataloader):
        if torch.cuda.device_count() > 0:
            data, target = data.cuda(), target.cuda()
        with torch.no_grad():
            output = model(data)
        loss += criterion(output, target).data.item() * len(data)
        _, predictions = output.data.topk(5, 1, True, True)
        topk_correct = predictions.eq(target.data.contiguous().view(len(data), 1).expand_as(predictions)).cpu()
        top1 += len(data) - topk_correct.narrow(1, 0, 1).sum().item()
        top5 += len(data) - topk_correct.sum().item()
        if (batch_idx + 1) % 10 == 0:
            processed_data = len(data) * (batch_idx + 1)
            print('Test set[{}/{}]: Top1: {:.2f}%, Top5: {:.2f}%, Average loss: {:.4f}, Average time cost: {:.3f} s'.format(
                processed_data, len(dataloader.dataset), 100 * top1 / processed_data,
                100 * top5 / processed_data, loss / processed_data, (time.time() - timer) / processed_data))

    loss /= len(dataloader.dataset)
    print('Test set[{}]: Top1: {:.2f}%, Top5: {:.2f}%, Average loss: {:.4f}, Average time cost: {:.3f} s'.format(
        len(dataloader.dataset), 100 * top1 / len(dataloader.dataset),
        100 * top5 / len(dataloader.dataset), loss, (time.time() - timer) / len(dataloader.dataset)))
