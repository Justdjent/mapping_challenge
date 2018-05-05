import torch
import numpy as np
import cv2
from torch.utils.data import Dataset
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import random
import os
import prepare_data
import os

data_path = Path('data')
data_directory = "data/"
annotation_file_template = "{}/{}/annotation{}.json"

TRAIN_IMAGES_DIRECTORY = "../mapping-challenge-starter-kit/data/train/images"

VAL_IMAGES_DIRECTORY = "../mapping-challenge-starter-kit/data/val/images"

class MapDataset(Dataset):
    def __init__(self, file_names: str, to_augment=False, transform=None, mode='train', problem_type=None):
        self.file_names = file_names
        self.to_augment = to_augment
        self.transform = transform
        self.mode = mode
        self.problem_type = problem_type
        ## !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        self.coco = COCO(self.file_names)
        self.image_ids = self.coco.getImgIds(catIds=self.coco.getCatIds())

    def __len__(self):
        return len(self.image_ids)
        # return 1

    def __getitem__(self, idx):
        # print(self.file_names)
        # print(idx)
        # print(self.file_names[idx], len(self.file_names), idx)
        img = self.coco.loadImgs(self.image_ids[idx])[0]
        annotation_ids = self.coco.getAnnIds(imgIds=img['id'])
        annotations = self.coco.loadAnns(annotation_ids)
        # img_file_name = self.file_names[idx]
        pic = load_image(img, self.mode)
        mask = load_mask(annotations, img)
        pic, mask = self.transform(pic, mask)

        if self.problem_type == 'binary':
            return to_float_tensor(pic),\
                   torch.from_numpy(np.expand_dims(mask, 0)).float()

        else:
            # return to_float_tensor(img), torch.from_numpy(mask).long()
            return to_float_tensor(img), to_float_tensor(mask)
        # else:
        #     return to_float_tensor(img)# , str(img_file_name)


def to_float_tensor(img):
    return torch.from_numpy(np.moveaxis(img, -1, 0)).float()


def load_image(img, mode):
    # path_ = "data/stage1_train_/{}/images/{}.png".format(path, path)
    # if mode != 'train':
    #     # path_ = "data/stage1_test/{}/images/{}.png".format(path, path)
    #     path_ = "data/cropped_test_2/{}".format(path)
    # if not os.path.isfile(path_):
    #     print('{} was empty'.format(path_))
    # img = cv2.imread(str(path_))
    if mode == 'valid':
        image_path = os.path.join(VAL_IMAGES_DIRECTORY, img["file_name"])
    else:
        image_path = os.path.join(TRAIN_IMAGES_DIRECTORY, img["file_name"])
    I = io.imread(image_path)
    # I1 = cv2.imread(image_path)
    # print(path_, img.shape)
    return I


def load_mask(annotations, img):

    mask = np.zeros((img['height'], img['width']))
    for i in annotations:
        rle = cocomask.frPyObjects(i['segmentation'], img['height'], img['width'])
        m = cocomask.decode(rle)
        # m.shape has a shape of (300, 300, 1)
        # so we first convert it to a shape of (300, 300)
        m = m.reshape((img['height'], img['width']))
        mask += m
    # path_ = "data/stage1_train_/{}/masksmask.png".format(path)
    # if mode != 'train':
    #     path_ = "data/stage2_test/{}/images/{}.png".format(path, path)
    # if not os.path.isfile(path_):
    #     print('{} was empty'.format(path_))
    # factor = prepare_data.binary_factor
    # mask = cv2.imread(str(path_))
    # kernel = np.ones((4, 4), np.uint8)
    # seed = cv2.erode(mask[:, :, 0], kernel, iterations=1)
    # border = mask[:, :, 0] - seed
    # mask[:, :, 1] = np.zeros(seed.shape)
    # mask[:, :, 1] = seed
    # mask[:, :, 2] = np.zeros(seed.shape)
    # mask[:, :, 2] = border

    return mask.astype(np.uint8)
