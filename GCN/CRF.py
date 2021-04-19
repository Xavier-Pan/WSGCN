import numpy as np
import torch
import scipy.misc
import os
import datetime
import time
import pydensecrf.densecrf as dcrf
import tqdm
import misc_old

from cv2 import imread
from torch import nn
from multiprocessing import Pool
from functools import partial
from pydensecrf.utils import unary_from_softmax
from utils import colors_map, load_img_name_list, show_timing
from utils import evaluate_dataset_IoU
from utils import get_least_modify_file
from config import opt as args
from PIL import Image


def crf_inference(img, probs, CRF_parameter, scale_factor=1, labels=21):
    h, w = img.shape[:2]
    n_labels = labels

    d = dcrf.DenseCRF2D(w, h, n_labels)
    pred_softmax = torch.nn.Softmax(dim=0)
    probs = pred_softmax(torch.tensor(probs)).numpy()
    unary = unary_from_softmax(probs)
    unary = np.ascontiguousarray(unary)

    d.setUnaryEnergy(unary)
    d.addPairwiseGaussian(sxy=CRF_parameter["pos_xy_std"] / scale_factor,
                          compat=CRF_parameter["pos_w"])
    d.addPairwiseBilateral(sxy=CRF_parameter["bi_xy_std"] / scale_factor,
                           srgb=CRF_parameter["bi_rgb_std"],
                           rgbim=np.copy(img),
                           compat=CRF_parameter["bi_w"])
    Q = d.inference(CRF_parameter["iter_max"])
    return np.array(Q).reshape((n_labels, h, w))


def crf(img_name, CRF_parameter, save_path_label, save_path_logit, img=None,
        probs=None, prediction_root=None, scale_factor=1, labels=21):

    if img is None:
        img = imread(os.path.join(args.path4Image, img_name + '.jpg'))
    H, W = img.shape[:2]
    # load predict_dict 
    if prediction_root is None:
        prediction_root = os.path.join("predict_result_matrix_visual_new",
                                       "250")
    prect_dict = np.load(os.path.join(prediction_root, img_name + '.npy'),
                         allow_pickle=True).item()

    def crf_inf(predicted_dict, name=None):
        v = np.array(list(predicted_dict.values()))
        img_path = os.path.join(args.path4Image, name + '.jpg')
        orig_img = np.asarray(Image.open(img_path))
        crf_score = crf_inference(orig_img, v, labels=v.shape[0],
                                  CRF_parameter=CRF_parameter)
        h, w = orig_img.shape[:2]
        crf_dict = dict()
        crf_score_np = np.zeros(shape=(args.num_class, h, w))
        for i, key in enumerate(predicted_dict.keys()):
            crf_score_np[key] = crf_score[i]
            crf_dict[key] = crf_score[i]
        return crf_score_np, crf_dict

    crf_resut, crf_dict = crf_inf(predicted_dict=prect_dict, name=img_name)
    
    # save crf logit
    if not os.path.exists(save_path_logit):
        os.makedirs(save_path_logit)
    np.save(os.path.join(save_path_logit, img_name + '.npy'), crf_dict)
    
    # save crf label
    if not os.path.exists(save_path_label):
        os.makedirs(save_path_label)
    misc_old.toimage(crf_resut.argmax(axis=0), cmin=0, cmax=255, 
                     pal=colors_map, mode="P").save(
                     os.path.join(save_path_label, img_name + '.png'))


def apply(**kwargs):
    parameter_dict = dict()
    t_start = time.time()
    time_now = datetime.datetime.today()
    time_now = "{}_{}_{}_{}h{}m".format(time_now.year, time_now.month,
                                        time_now.day, time_now.hour,
                                        time_now.minute)
    descript = ""
    parameter_dict["num_cpu"] = os.cpu_count()//2
    parameter_dict["CRF_parameter"] = args.CRF
    parameter_dict["path4saveCRF_label"] = args.path4Complete_label_label
    parameter_dict["path4saveCRF_logit"] = args.path4Complete_label_logit
    if "pred_root" not in kwargs.keys():
        parameter_dict["pred_root"] = args.path4GCN_logit
    else:
        parameter_dict["pred_root"] = kwargs["pred_root"]
    
    parameter_dict["f_list"] = args.path4train_images

    evaluate_folder = parameter_dict["path4saveCRF_label"]
    img_list = load_img_name_list(parameter_dict["f_list"])
    # === load parameter
    for k, v in kwargs.items():
        if k in parameter_dict.keys():
            if "CRF_parameter" == k:
                parameter_dict[k] = eval(v)
            else:
                parameter_dict[k] = v
            print("{}: {}".format(k, parameter_dict[k]))

    print("path4saveCRF_label: ", parameter_dict["path4saveCRF_label"])
    print("pred_root: ", parameter_dict["pred_root"])

    p = Pool(parameter_dict["num_cpu"])
    crfP = partial(crf,
                   prediction_root=parameter_dict["pred_root"],
                   save_path_label=parameter_dict["path4saveCRF_label"],
                   save_path_logit=parameter_dict["path4saveCRF_logit"],
                   CRF_parameter=parameter_dict["CRF_parameter"])
    # run crf by multiprocessing
    for _ in tqdm.tqdm(p.imap_unordered(crfP, img_list), total=len(img_list)):
        pass
    p.close()
    p.join()
    evaluate_dataset_IoU(file_list=parameter_dict["f_list"],
                         predicted_folder=evaluate_folder,
                         descript=descript,
                         path4GT=args.path4VOC_class_aug)
    show_timing(time_start=t_start, time_end=time.time())


if __name__ == "__main__":
    apply()

