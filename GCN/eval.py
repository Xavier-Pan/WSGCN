import numpy as np
import scipy.sparse as sp
import torch

import pickle as pkl
import networkx as nx
import torch.nn.functional as F
from torch import nn
from scipy.sparse.linalg.eigen.arpack import eigsh
import scipy.misc
import sys
import os
from config import opt as args
from PIL import Image
import datetime
import fire
import time
import getpass
from cv2 import imread, imwrite
import pydensecrf.densecrf as dcrf


SEG_LIST = [
    'BACKGROUND', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
SEG_ID_TO_NAME = dict(zip(np.arange(len(SEG_LIST)), SEG_LIST))


def load_img_name_list(dataset_path):
    """
    return imgs_list e.g.  imgs_list[0] = 2007_000121
    """
    img_gt_name_list = open(dataset_path).read().splitlines()
    img_name_list = [
        img_gt_name.split(' ')[0][-15:-4] for img_gt_name in img_gt_name_list
    ]
    """ /JPEGImages/2007_000121.jpg -> [-15:-4] = 2007_000121 """
    return img_name_list


class IOUMetric:
    """
    Class to calculate mean-iou using fast_hist method
    IoU = IOUMetric(num_class)
    IoU.addbatch(predictions,groundtruths)
    acc, acc_cls, iu, mean_iu, fwavacc = IoU.evaluate()
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
        self.num_pixel4dataset = 0
        self.num_train_pixel4dataset = 0

    def _fast_hist(self, label_pred, label_true):
        # ignore 255 and negative value for ground truth
        mask = (label_true >= 0) & (label_true < self.num_classes)
        # ignore 255 and negative value for prediction
        mask = mask & (label_pred >= 0) & (label_pred < self.num_classes)
        self.num_train_pixel4dataset += np.sum(mask)
        self.num_pixel4dataset += mask.size
        hist = np.bincount(
            self.num_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.num_classes**2).reshape(self.num_classes,
                                                   self.num_classes)

        return hist

    def add_batch(self, predictions, gts):
        for lp, lt in zip(predictions, gts):
            self.hist += self._fast_hist(lp.flatten(), lt.flatten())

    def evaluate(self):
        """
        iu: [num_classes,] is a numpy array. each item is IoU for class_i
        mean_iu_tensor: a tensor, take the average of iu
        acc_cls: accuracy for each class
        """
        acc = np.diag(self.hist).sum() / self.hist.sum()
        acc_cls = np.diag(self.hist) / self.hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(self.hist) / (self.hist.sum(axis=1) +
                                   self.hist.sum(axis=0) - np.diag(self.hist))
        # nanmean just ignore nan in the item
        mean_iu_tensor = torch.from_numpy(np.asarray(np.nanmean(iu)))
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu_tensor, fwavacc


def evaluate_dataset_IoU(predicted_folder="../Data/GCN4DeepLab/Label/",
                         path4GT="../Data/VOC12/VOC2012/SegmentationClassAug/",
                         file_list=None,
                         ignore_img_list=[],
                         save_info=True,
                         descript=None):

    # file_list
    img_list = load_img_name_list("../Data/VOC12/train.txt")
    #os.listdir(predicted_folder)
    IoU = IOUMetric(args.num_class)
    num_imgs = len(img_list)
    i = 0
    mask_predit_batch = []
    mask_GT_batch = []
    for img_name in img_list:
        if img_name in ignore_img_list:
            print("ignore: {}".format(img_name))
            continue
        i = i + 1
        print("[{}/{}]evaluate: ".format(i, num_imgs), img_name, end='\r')
        mask_gt = Image.open(os.path.join(path4GT, img_name+'.png'))
        mask_gt = np.asarray(mask_gt)
        #try:
        mask_predit = Image.open(
                os.path.join(predicted_folder, img_name+'.png'))
        #except:
        #    continue
        mask_predit = np.asarray(mask_predit)
        mask_predit_batch.append(mask_predit)
        mask_GT_batch.append(mask_gt)
    IoU.add_batch(mask_predit_batch, mask_GT_batch)
    acc, acc_cls, iu, mean_iu_tensor, fwavacc = IoU.evaluate()

    # show information
    print("pseudo pixel label ratio: {:>5.2f} %".format(
        IoU.num_train_pixel4dataset / IoU.num_pixel4dataset * 100))
    # show IoU of each class
    print("=" * 34)
    for idx, iu_class in enumerate(iu):
        print("{:12}: {:>17.2f} %".format(SEG_ID_TO_NAME[idx], iu_class * 100))
    print("=" * 34)
    print("IoU:{:>27.2f} %  Acc:{:>13.2f} %".format(mean_iu_tensor * 100,
                                                    acc * 100))
    print("=" * 34)

    return mean_iu_tensor.item(), acc


if __name__ == "__main__":
    evaluate_dataset_IoU()
