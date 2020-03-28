import torch
from torch.utils import data
from torchvision.datasets import CocoDetection
from pycocotools.coco import COCO
from PIL import Image
import numpy as np
import cv2
import os

dataDir = '/media/WINDOW/Users/ari/Desktop/coco'
dataTypeTrain = 'train2014'
imgDirTrain = '{}/images/{}/'.format(dataDir, dataTypeTrain)
annFileTrain = '{}/annotations/instances_{}.json'.format(dataDir, dataTypeTrain)
dataTypeVal = 'val2014'
imgDirVal = '{}/images/{}/'.format(dataDir, dataTypeVal)
annFileVal = '{}/annotations/instances_{}.json'.format(dataDir, dataTypeVal)
labelFileDir = ROOT_DIR = os.path.dirname(os.path.realpath(__file__)) + '/coco_labels.txt'

def get_label_map(label_file):
    label_map = {}
    labels = open(label_file, 'r')
    for line in labels:
        ids = line.split(',')
        label_map[int(ids[0])] = int(ids[1])
    return label_map

class CocoSubsetDetection(data.Dataset):
    """
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        catNames (list): List subset of COCO
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, annFile, catNames=['person'], transform=None, target_transform=None):
        self.coco = COCO(annFile)
        catIds = self.coco.getCatIds(catNms=catNames)
        imgIds = self.coco.getImgIds(catIds=catIds)
        self.ids = list(sorted(imgIds))
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)

        path = self.coco.loadImgs(img_id)[0]['file_name']
        image = cv2.imread(os.path.join(self.root, path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes, labels = get_annotation(anns)

        if self.transform is not None:
            image, boxes, labels = self.transform(image, boxes, labels)

        if self.target_transform is not None:
            boxes, labels = self.target_transform(boxes, labels)

        return image, boxes, labels

def get_annotation(anns):
    boxes = []
    labels = []

    for anotation in anns:
        if (anotation['category_id']==1):
            labels.append(anotation['category_id'])
            [x, y, width, height] = anotation['bbox']
            boxes.append([x, y, x + width, y + height])

    return np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int64)
