import torch
import torch.nn.functional as F
import os
import scipy.sparse as sp
import numpy as np
import time

from PIL import Image
from torchvision import transforms as T
from torch.utils import data
from cv2 import imread
from config import opt as args
from utils import load_img_name_list


def normalize_t(mx):
    """Row-normalize sparse matrix in tensor"""
    rowsum = torch.sum(mx, dim=1)
    r_inv = torch.pow(rowsum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diagflat(r_inv)
    mx = torch.mm(r_mat_inv, mx)
    return mx


def preprocess_adj(aff_mat):
    adjT = torch.t(aff_mat)
    adj = torch.stack([aff_mat, adjT])
    adj, _ = adj.max(dim=0)
    return normalize_t(adj + torch.eye(adj.shape[0]))


class graph_voc(data.Dataset):
    def __init__(self, root=args.path4Image, graph_type="AFF", start_idx=0,
                 end_idx=None, device=None):
        self.label_list = load_img_name_list(args.path4train_images)
        self.seg_label_dict = dict()
        # AFF
        self.graph_type = graph_type  
        self.train_file = load_img_name_list(args.path4train_images)
        self.start_idx = start_idx
        self.end_idx = len(self.label_list) if end_idx is None else end_idx
        self.device = device
        print("self.device: ", self.device)
        self.ignore_list = []

    def load_data(self, graph_type='AFF', path=None, img_name=None,
                  path4Data=None, load_adjacency_mat=True):
        """
        return adj, features, labels, idx_train, idx_test, rgbxy, img_name
        adj: sparse matrix
        """
        t_start = time.time()
        graph = np.load(os.path.join(args.path4AffGraph, img_name + ".npy"))
        adj = preprocess_adj(torch.FloatTensor(graph))
        labels = Image.open(
            os.path.join(args.path4partial_label_label, img_name + '.png'))
        labels = np.asarray(labels)
        labels = np.reshape(labels, (-1)).astype(np.int16)

        # np.int8 turns 255 to -1
        labels = np.where(labels == -1, 255,
                          labels)
        # split foreground and background label
        label_fg = labels.copy()
        label_fg[label_fg == 0] = 255

        label_bg = labels.copy()
        label_bg[label_bg != 0] = 255

        # to tensor
        labels = torch.LongTensor(labels)
        label_fg_t = torch.LongTensor(label_fg)
        label_bg_t = torch.LongTensor(label_bg)

        img = imread(os.path.join(args.path4Image, img_name + ".jpg"))
        H_origin, W_origin, C = img.shape
        H = int(np.ceil(H_origin / 4))
        W = int(np.ceil(W_origin / 4))

        f_aff = np.load(os.path.join(args.path4node_feat, img_name + ".npy"))
        f_aff = np.squeeze(f_aff)
        f_aff = np.reshape(f_aff, (np.shape(f_aff)[0], H * W))
        allx = np.transpose(f_aff, [1, 0])
        feat = torch.FloatTensor(np.array(allx))
        # get rgb    
        img_dn = Image.fromarray(img).resize((W, H), Image.LANCZOS)
        img_dn = np.asarray(img_dn)
        rgbxy = np.zeros(shape=(H, W, 5))
        rgbxy[:, :, :3] = img_dn / 255.

        # get xy
        for i in range(H):
            for j in range(W):
                rgbxy[i, j, 3] = float(i) # / H
                rgbxy[i, j, 4] = float(j) # / W

        rgbxy_t = torch.FloatTensor(rgbxy)
        return {"adj_t": adj, "features_t": feat, "labels_t": labels,
            "rgbxy_t": rgbxy_t, "img_name": img_name, 
            "label_fg_t": label_fg_t, "label_bg_t": label_bg_t}

    def __getitem__(self, index):
        """
        return adj, feat, labels, idx_train_t, rgbxy, img_name, label_fg_t, label_bg_t
        """

        img_name = self.train_file[index]
        if self.start_idx <= index < self.end_idx:
            if img_name in self.ignore_list:
                print("[{}] ignore: {}".format(index, img_name))
                return None
            return self.load_data(graph_type=self.graph_type, 
                                  path=args.path4AffGraph, img_name=img_name,
                                  path4Data=args.path4Data)
        else:
            return None

    def __len__(self):
        return len(self.train_file)
