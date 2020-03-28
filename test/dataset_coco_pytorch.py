"""
https://karanchahal.github.io/2018/05/25/Implementing-Object-Detectors-Input-Pipeline/
"""

from pycocotools.coco import COCO
from torch.utils.data import Dataset
import torch
import itertools

dataDir='/media/WINDOW/Users/ari/Desktop/coco'
dataType='train2014'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
coco = COCO(annFile)

class CocoDatasetObjectDetection(Dataset):
    """Object detection dataset."""

    def __init__(self, imgIds, coco, transform=None):
        """
        Args:
            imgIds (string): image ids of the images in COCO.
            coco (object): Coco image data helper object.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.coco = coco
        self.imageIds = imgIds
        self.transform = transform
        # self.bounding_box_mode = bounding_box_mode

    def __len__(self):
        return len(self.imageIds)

    def __getitem__(self, idx):
        img_id = self.imageIds[idx]
        image_node = self.coco.loadImgs(img_id)[0]

        coordsList = []
        annIds = coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anns = coco.loadAnns(annIds)
        for a in anns:
            coordsList.append(a['bbox'])

        return image_node, coordsList

def load_bounding_box(image_id):
    coordsList = []
    annIds = coco.getAnnIds(imgIds=image_id, iscrowd=None)
    anns = coco.loadAnns(annIds)
    for a in anns:
        coordsList.append(a['bbox'])
    print(coordsList)

def coco_dataloader(cat_name):
    catIds = coco.getCatIds([cat_name]);
    personIds = coco.getImgIds(catIds=catIds);
    train_dataset = CocoDatasetObjectDetection(imgIds=personIds, coco=coco)

    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=512, shuffle=True, num_workers=2, pin_memory=False)

    for i, img_nodes in enumerate(train_dataloader):
        # Image node inspection
        if (i == 0):
            # license, file_name, coco_url, height, width, date_captured, flickr_url, id
            for key, value in img_nodes[0].items() :
                print(key)

            # Bounding box
            print('\nlen(img_nodes[1]): {}'.format(len(img_nodes[1])))
            for j in range(len(img_nodes[1])):
                print('len(img_nodes[1][j]): {}'.format(len(img_nodes[1][j])))

def getAnchorsForPixelPoint(i, j, width, height):
    anchors = []
    scales = [32,64,128,256]
    aspect_ratios = [[1,1]]
    for ratio in aspect_ratios:
        x = ratio[0]
        y = ratio[1]
        x1 = i
        y1 = j

        for scale in scales:
            w = x*(scale)
            h = y*(scale)
            anchors.append([x1,y1,w,h])
    return anchors

if __name__ == "__main__":
    # Load bounding box of image
    # load_bounding_box(129586)

    # Dataloader COCO subset person + bounding boxs
    coco_dataloader('person')

    # Anchors box test
    # for j, i in itertools.product(range(300), repeat=2):
    #     anchors = getAnchorsForPixelPoint(i, j, 300, 300)
    # print(anchors)
