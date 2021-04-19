import numpy as np
import torch
import torch.nn.functional as F
import scipy.misc
import os
import datetime
import fire
import time
import pydensecrf.densecrf as dcrf

from torch import nn
from config import opt as args
from PIL import Image
from cv2 import imread
from tensorboardX import SummaryWriter
from pydensecrf.utils import unary_from_softmax
from xml.dom import minidom


# color maps for VOC2012 dataset
colors_map = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
              [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
              [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
              [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
              [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128],
              [0, 0, 255]]
ANNOT_FOLDER_NAME = "Annotations"
SEG_LIST = [
    'BACKGROUND', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

# {class name:id}
SEG_NAME_TO_ID = dict(zip(SEG_LIST, range(len(SEG_LIST))))
SEG_ID_TO_NAME = dict(zip(np.arange(len(SEG_LIST)), SEG_LIST))

# {id:class name}
CLS_NAME_TO_ID = dict(zip(SEG_LIST[1:], range(len(SEG_LIST[1:]))))
CLS_ID_TO_NAME = dict(zip(range(len(SEG_LIST[1:])), SEG_LIST[1:]))


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {
        c: np.identity(len(classes))[i, :]
        for i, c in enumerate(classes)
    }
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    # Convert a scipy sparse matrix to a torch sparse tensor
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class IOUMetric:
    """
    calculate mIoU through fast_hist method
    acc, acc_cls, iu, mean_iu, fwavacc = IoU.evaluate()
    """
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))
        self.num_pixel4dataset = 0
        self.num_train_pixel4dataset = 0

    def _fast_hist(self, label_pred, label_true):
        # ignore 255 and negative value
        mask = (label_true >= 0) & (label_true < self.num_classes)
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
        # ignore nan in the item
        mean_iu_tensor = torch.from_numpy(np.asarray(np.nanmean(iu)))
        freq = self.hist.sum(axis=1) / self.hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        return acc, acc_cls, iu, mean_iu_tensor, fwavacc


def load_img_name_list(dataset_path):
    img_gt_name_list = open(dataset_path).read().splitlines()
    img_name_list = [
        img_gt_name.split(' ')[0][-15:-4] for img_gt_name in img_gt_name_list
    ]
    return img_name_list


def evaluate_dataset_IoU(predicted_folder=None, path4GT=args.path4VOC_class_aug,
                         file_list=None, ignore_img_list=[], save_info=True,
                         descript=None):
    if predicted_folder is None:
        predicted_folder = os.path.join(
            args.path4GCN_label, get_least_modify_file(args.path4GCN_label))

    if file_list is None:
        file_list = args.path4train_images
    

    print("=" * 40)
    print("file_list ", file_list)
    print("predicted_folder:", predicted_folder)
    img_list = load_img_name_list(file_list)
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
        mask_gt = Image.open(os.path.join(path4GT, img_name + '.png'))
        mask_gt = np.asarray(mask_gt)
        mask_predit = Image.open(
            os.path.join(predicted_folder, img_name + '.png'))
        mask_predit = np.asarray(mask_predit)
        # upsampling
        if mask_predit.shape[0] < mask_gt.shape[0] or mask_predit.shape[
                1] < mask_gt.shape[1]:
            mask_predit_up = Image.fromarray(mask_predit).resize(
                (mask_gt.shape[1], mask_gt.shape[0]), Image.NEAREST)
            mask_predit = np.asarray(mask_predit_up)
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
    # save information in information_for_IoU.md
    if descript is not None:
        time_now = datetime.datetime.today()
        time_now = "{}-{}-{}  {}:{}".format(time_now.year, time_now.month,
                                            time_now.day, time_now.hour,
                                            time_now.minute)

        if not os.path.isfile("meanIoU.md"):
            f = open("meanIoU.md", "w")
            f.close()

        with open("meanIoU.md", "r") as f:
            old_context = f.read()
        with open("meanIoU.md", "r+") as f:
            f.write("{}  \n".format(time_now))
            f.write("---\n")
            f.write("\n|Setting|Value|\n")
            f.write("|-|-|\n")
            f.write("**predicted folder**|{}  \n".format(predicted_folder))
            f.write("**Dataset files** |{}  \n".format(
                os.path.basename(args.path4Data)))
            f.write("**AFF_path**| {}  \n".format(
                os.path.basename(args.path4AffGraph)))
            f.write("**apply CRF**  \n")
            f.write("**epoch**| {}  \n".format(args.max_epoch))
            f.write("**hid unit**| {}  \n".format(args.num_hid_unit))
            f.write("**drop out**| {}  \n".format(args.drop_rate))
            f.write("**stript**| {}  \n".format(descript))
            f.write("-" * 3)
            f.write("\n|Class|IoU|\n")
            f.write("|-|-|\n")
            for idx, iu_class in enumerate(iu):
                f.write("{:12}| {:>17.2f} %  \n".format(
                    SEG_ID_TO_NAME[idx], iu_class * 100))
            f.write("-" * 3)
            f.write("\n|pseudo pixel label ratio|Acc|meanIoU|\n")
            f.write("|-|-|-|\n")
            f.write("|{:<5.2f} %|{:>5.2f} % | {:>27.2f} % \n".format(
                IoU.num_train_pixel4dataset / IoU.num_pixel4dataset * 100,
                acc * 100,
                mean_iu_tensor.item() * 100))
            f.write("-" * 3 + "\n")
            f.write(old_context)

    return mean_iu_tensor.item(), acc


def crf_inference(img, probs, t=10, scale_factor=1, labels=21):
    """
    dense crf
    - img: np_array [h,w,c]
    - probs: prediction_score [c,h,w]
    - t: number of iteration for inference
    """
    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)

    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    img_c = np.ascontiguousarray(img)

    d.setUnaryEnergy(unary)

    d.addPairwiseGaussian(sxy=4 / scale_factor, compat=3)
    d.addPairwiseBilateral(sxy=20 / scale_factor,
                           srgb=3,
                           rgbim=np.copy(img_c),
                           compat=10)
    Q = d.inference(t)

    return np.array(Q).reshape((n_labels, h, w))


def compute_joint_loss(ori_img, seg, seg_label, croppings, critersion,
                       DenseEnergyLosslayer):
    """
    1. seg_label: pseudo label for segmentation
    2. seg: ouput of seg_model (b,c,w,h)
    ---
    seperate bg_loss and fg_loss
    return cross-entropy loss, dense loss
    """
    seg_label = np.expand_dims(
        seg_label, axis=1
    ) 
    seg_label = torch.from_numpy(seg_label) 

    w = seg_label.shape[2]
    h = seg_label.shape[3]
    pred = F.interpolate(seg, (w, h), mode="bilinear", align_corners=False)

    # apply softmax to model prediction
    pred_softmax = torch.nn.Softmax(dim=1)
    pred_probs = pred_softmax(pred)

    ori_img = torch.from_numpy(ori_img.astype(np.float32))
    croppings = torch.from_numpy(
        croppings.astype(np.float32).transpose(2, 0, 1))

    seg_label_tensor = seg_label.long().cuda()

    seg_label_copy = torch.squeeze(seg_label_tensor.clone())
    bg_label = seg_label_copy.clone()
    fg_label = seg_label_copy.clone()

    # ignore label (255) for foreground and background
    bg_label[seg_label_copy != 0] = 255
    fg_label[seg_label_copy == 0] = 255

    bg_celoss = critersion(pred, bg_label.long().cuda())
    fg_celoss = critersion(pred, fg_label.long().cuda())

    # dense loss
    dloss = DenseEnergyLosslayer(ori_img, pred_probs, croppings, seg_label)
    dloss = dloss.cuda()

    celoss = bg_celoss + fg_celoss

    return celoss, dloss


class Normalize():
    def __init__(self, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):

        self.mean = mean
        self.std = std

    def __call__(self, img, ori_img, croppings):
        imgarr = np.asarray(img)
        proc_img = np.empty_like(imgarr, np.float32)

        proc_img[..., 0] = (imgarr[..., 0] / 255. - self.mean[0]) / self.std[0]
        proc_img[..., 1] = (imgarr[..., 1] / 255. - self.mean[1]) / self.std[1]
        proc_img[..., 2] = (imgarr[..., 2] / 255. - self.mean[2]) / self.std[2]
        croppings = np.ones_like(imgarr)
        return proc_img, imgarr, croppings


def _crf_with_alpha(ori_img, cam_dict, alpha=32, use_crf=True):
    v = np.array(list(cam_dict.values()))
    bg_score = np.power(1 - np.max(v, axis=0, keepdims=True), alpha)
    bgcam_score = np.concatenate((bg_score, v), axis=0)
    if use_crf:
        crf_score = crf_inference(ori_img,
                                  bgcam_score,
                                  labels=bgcam_score.shape[0])
        bgcam_score = crf_score

    n_crf_al = np.zeros([args.num_class, bg_score.shape[1], bg_score.shape[2]])

    n_crf_al[0, :, :] = bgcam_score[0, :, :]

    for i, key in enumerate(cam_dict.keys()):
        n_crf_al[key + 1] = bgcam_score[i + 1]

    return n_crf_al


def load_image_label_from_xml(img_name, voc12_root):
    """
    no background index
    ===
    - img_name
    - return np array 
    """
    el_list = minidom.parse(
        os.path.join(voc12_root, ANNOT_FOLDER_NAME,
                     img_name + '.xml')).getElementsByTagName('name')

    multi_cls_lab = np.zeros((20), np.float32)

    for el in el_list:
        cat_name = el.firstChild.data
        if cat_name in SEG_LIST:
            cat_num = CLS_NAME_TO_ID[cat_name]
            multi_cls_lab[cat_num] = 1.0

    return multi_cls_lab


def compute_seg_label(ori_img,
                      cam_label,
                      norm_cam,
                      threshold4conf=0.8,
                      use_crf=True,
                      confident_region=1.):
    """
    norm_cam: value between 0,1]
    let seg_score_np = CRF(ha_CAM+la_CAM)
    return 
        - mask of seg_score with ignore region
        - seg_score_np
    """
    cam_label = cam_label.astype(np.uint8)
    cam_dict = {}
    cam_np = np.zeros_like(norm_cam)
    for i in range(20):
        if cam_label[i] > 1e-5:
            cam_dict[i] = norm_cam[i]
            cam_np[i] = norm_cam[i]

    # confident background 
    bg_score = np.power(1 - np.max(cam_np, 0), 32) 
    bg_score = np.expand_dims(bg_score, axis=0)
    cam_all = np.concatenate((bg_score, cam_np))
    _, bg_w, bg_h = bg_score.shape

    cam_img = np.argmax(cam_all, 0)

    # crf with condident foreground
    crf_la = _crf_with_alpha(ori_img, cam_dict, 4, use_crf=use_crf)
    # crf with condident background
    crf_ha = _crf_with_alpha(ori_img, cam_dict, 32, use_crf=use_crf)
    crf_la_label = np.argmax(crf_la, 0)
    crf_ha_label = np.argmax(crf_ha, 0)
    crf_label = crf_la_label.copy()
    # ignore low alpha background label
    crf_label[crf_la_label == 0] = 255
    # adopt high alpha background label 
    crf_label[crf_ha_label == 0] = 0

    # find the top k% score region as confident region and then union all confident region
    single_img_classes = np.unique(crf_la_label)
    cam_sure_region = np.zeros([bg_w, bg_h], dtype=bool)
    for class_i in single_img_classes:
        # foreground 
        if class_i != 0:
            class_not_region = (cam_img != class_i)  
            cam_class = cam_all[class_i, :, :]
            # set unconfident region to 0
            cam_class[class_not_region] = 0 
            cam_class_order = cam_class[cam_class > 0.1]  

            cam_class_order = np.sort(cam_class_order)
            # threshold
            confidence_pos = int(cam_class_order.shape[0] * (1. - confident_region))
            confidence_value = cam_class_order[confidence_pos]

            # keep the value which higher than the threshold
            class_sure_region = (cam_class > confidence_value)
            # expand the confident region
            cam_sure_region = np.logical_or(cam_sure_region, class_sure_region)
        else:  # background
            # find foreground
            class_not_region = (cam_img != class_i)
            cam_class = cam_all[class_i, :, :]
            # set foreground region score to 0
            cam_class[class_not_region] = 0
            # only confident background region would be take
            class_sure_region = (cam_class > threshold4conf)
            # expand the sure region
            cam_sure_region = np.logical_or(cam_sure_region, class_sure_region)

    cam_not_sure_region = ~cam_sure_region

    # take low alpha foreground score and high alpha background score
    crf_label_np = np.concatenate(
        [np.expand_dims(crf_ha[0, :, :], axis=0), crf_la[1:, :, :]])

    crf_not_sure_region = np.max(crf_label_np, 0) < threshold4conf
    not_sure_region = np.logical_or(crf_not_sure_region, cam_not_sure_region)

    crf_label[not_sure_region] = 255

    return crf_label, crf_label_np


def gen_label(num_class=20, save_img=True, voc_imgs_root=args.path4Image,
              predict_root=args.path4CAM, destination=args.path4IRN4GCN,
              use_crf=False, save_npy=False):
    destination_np = destination + "_SCORE"
    imgs = [
        os.path.splitext(f)[0] for f in os.listdir(predict_root)
        if os.path.splitext(f)[-1] == '.npy'
    ]
    voc_imgs = [
        os.path.join(voc_imgs_root, f) for f in os.listdir(voc_imgs_root)
        if os.path.splitext(f)[0] in imgs
    ]

    if not os.path.exists(destination):
        os.mkdir(destination)
    if not os.path.exists(destination_np):
        os.mkdir(destination_np)
    lenth = len(imgs)
    cam_exist_list = [os.path.splitext(f)[0] for f in os.listdir(destination)]
    for idx, (cam_file, img_name) in enumerate(zip(imgs, voc_imgs)):
        if cam_file in cam_exist_list:
            print("[{}/{}]{} already exist!!".format(idx + 1, lenth, cam_file))
            continue
        img = imread(img_name)
        (H, W) = img.shape[:2]
        cams = np.zeros((num_class, H, W))
        dict_np = np.load(os.path.join(predict_root, cam_file + '.npy')).item()
        print('[{idx}/{lenth}] cam_file:{cam_file}  cams.shape: {cams_shape}'.
              format(idx=idx + 1,
                     lenth=lenth,
                     cam_file=cam_file,
                     cams_shape=cams.shape))
        cam_label = np.zeros(20)
        cam_dict = {}
        cam_temp = None
        for key, cam in dict_np.items():
            cam = F.interpolate(torch.tensor(
                cam[np.newaxis, np.newaxis, :, :]), (H, W),
                                mode="bilinear",
                                align_corners=False).numpy()
            cams[key] = cam  # note! cam label from 0-29
            cam_temp = cam.copy()
            cam_label[key] = 1
            cam_dict[key] = cams[key]

        print(
            '[{idx}/{lenth}] cam_file:{cam_file}  cams.shape: {cams_shape}  cam.shape: {cam_shape}'
            .format(idx=idx + 1,
                    lenth=lenth,
                    cam_file=cam_file,
                    cams_shape=cams.shape,
                    cam_shape=cam_temp.shape))

        cam_label = cam_label.astype(np.uint8)
        seg_label_crf_conf, seg_score_crf_conf = compute_seg_label(
            ori_img=img, cam_label=cam_label, norm_cam=cams, use_crf=use_crf)

        # save label score in dictionary type
        pseudo_label_dict = dict()
        img_label = load_image_label_from_xml(cam_file)
        pseudo_label_dict[0] = seg_score_crf_conf[0]
        for key in img_label:
            pseudo_label_dict[int(key)] = seg_score_crf_conf[int(key + 1)]

        if save_npy:
            np.save(os.path.join(destination_np, "{}".format(cam_file)),
                    pseudo_label_dict)
        # save label mask
        scipy.misc.toimage(seg_label_crf_conf,
                           cmin=0,
                           cmax=255,
                           pal=colors_map,
                           mode='P').save(
                               os.path.join(destination, cam_file + '.png'))


def show_timing(time_start, time_end, show=False):
    # show timimg in the format: h m s
    time_hms = "Total time elapsed: {:.0f} h {:.0f} m {:.0f} s".format(
        (time_end - time_start) // 3600, (time_end - time_start) / 60 % 60,
        (time_end - time_start) % 60)
    if show:
        print(time_hms)
    return time_hms


def visulize_cam(root, destination="VGG_CAM_CRF", alpha=16, use_crf=False):
    # transfer CAMs into pseudo label
    
    destination = "{}_HA{}".format(destination, alpha)
    if not os.path.exists(destination):
        os.mkdir(destination)
    cam_path = root
    train_list = np.array(load_img_name_list(args.path4train_images))
    len_list = train_list.size
    for idx, name in enumerate(train_list):
        print("[{}/{}] {}...".format(idx + 1, len_list, name), end='\r')
        cam_dict = np.load(os.path.join(cam_path, name + '.npy'),
                           allow_pickle=True).item()
        img = imread(os.path.join(args.path4Image, name + '.jpg'))
        pseudo_label = _crf_with_alpha(ori_img=img,
                                       cam_dict=cam_dict,
                                       alpha=alpha,
                                       use_crf=use_crf)
        scipy.misc.toimage(np.argmax(pseudo_label, axis=0), cmin=0, cmax=255,
                           pal=colors_map, mode='P').save(
                           os.path.join(destination, name + '.png'))

    print("spend time:{:.1f}s".format(time.time() - t_start))


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x, gt, ignore_index=255):
        b = -torch.exp(x) * x
        b = b[gt == 255].mean()
        return b


def get_least_modify_file(folder, show_sorted_data=False):
    dir_list = os.listdir(folder)
    dir_list = sorted(dir_list,
                      key=lambda x: os.path.getmtime(os.path.join(folder, x)))
    print(dir_list)
    if show_sorted_data:
        for it in dir_list:
            print("dir_list: {:<80}  time: {:}".format(
                it, os.path.getmtime(os.path.join(args.path4GCN_logit, it))))
    return dir_list[-1]


def normalize_t(mx):
    """
    row-normalize sparse matrix in tensor
    ---
    - mx: 2D-tensor
    """
    rowsum = torch.sum(mx, dim=1)
    r_inv = torch.pow(rowsum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    r_mat_inv = torch.diagflat(r_inv)
    mx = torch.mm(r_mat_inv, mx)
    return mx


def preprocess_adj(aff_mat, device):
    adjT = torch.t(aff_mat)
    adj = torch.stack([aff_mat, adjT])
    adj, _ = adj.max(dim=0)
    return normalize_t(adj + torch.eye(adj.shape[0]).to(device))

    
if __name__ == "__main__":
    fire.Fire()
