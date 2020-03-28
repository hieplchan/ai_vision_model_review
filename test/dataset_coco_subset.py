import sys
sys.path.append('..')

from PIL import ImageDraw
import torch
import torchvision.transforms as transforms
import time

from utils.datasets.coco_dataset import CocoSubsetDetection
from utils.misc import Timer
from models import DEVICE

timer = Timer()

dataDir = '/media/WINDOW/Users/ari/Desktop/coco'
train_dataType = 'train2014'
train_annFile = '{}/annotations/instances_{}.json'.format(dataDir, train_dataType)
train_imgDir = '{}/images/{}/'.format(dataDir, train_dataType)
val_dataType = 'val2014'
val_annFile = '{}/annotations/instances_{}.json'.format(dataDir, val_dataType)
val_imgDir = '{}/images/{}/'.format(dataDir, val_dataType)

subset_names = ['person']
transform = transforms.Compose([transforms.Resize((300, 300), interpolation=2),
                                transforms.ToTensor()])
coco_train = CocoSubsetDetection(root=train_imgDir, annFile=train_annFile, catNames=subset_names, transform=transform)
# coco_val = CocoSubsetDetection(root=val_imgDir, annFile=val_annFile, catNames=subset_names, transform=None)

if __name__ == "__main__":
    print('Training subset {} has total {} images'.format(subset_names, coco_train.__len__()))
    # print('Validating subset {} has total {} images'.format(subset_names, coco_val.__len__()))

    """
    COCO subclass visualize
    """
    # item[0] is image, item[1] is annotations
    # data_node = coco_train.__getitem__(0)
    # source_img = data_node[0]
    # for anotation in data_node[1]:
    #     # Person category_id = 1
    #     if (anotation['category_id'] == 1):
    #         [x, y, width, height] = anotation['bbox']
    #         # print([x, y, width, height])
    #         draw = ImageDraw.Draw(source_img)
    #         draw.rectangle(((x, y, x + width, y + height)), fill=None, outline="red")
    # source_img.save('test.jpg')

    """
    COCO Dataset DataLoader
    """
    train_dataloader = torch.utils.data.DataLoader(coco_train, batch_size=512, shuffle=True, num_workers=5, pin_memory=True)

    timer.start(key='DataLoader')

    for idx, nodes in enumerate(train_dataloader):
        print(idx)
        print(nodes.shape)

    interval = timer.end(key='DataLoader')
