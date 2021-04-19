import numpy as np
import pickle
import os
import torch
import torch.nn.functional as F
import tqdm
import scipy.misc
import misc_old
import time
import sys

from PIL import Image
from cv2 import imread
from scipy import sparse
from scipy.sparse import csr_matrix
from utils import show_timing
from utils import load_img_name_list, colors_map, compute_seg_label
from utils import evaluate_dataset_IoU, load_image_label_from_xml
from multiprocessing import Pool
from functools import partial
from train import compute_lap_test
from config import opt as args


def gen_partial_label_with_ratio(img_name,
                                 predict_root=None,
                                 destination4visulization=None,
                                 destination4logit=None,
                                 confident_region=1.,
                                 show_infromation=False,
                                 device="0"):
    """
    generate partial pseudo label through threshold mechanism
    ---
    the data folder will be generated:
    - destination4visulization + '_DN_UP'
    - destination4visulization + '_DN'
    - destination4visulization + '_UP'   
    """

    img = imread(os.path.join(args.path4Image, img_name + ".jpg"))
    H_origin, W_origin, C = img.shape
    H = int(np.ceil(H_origin / 4))
    W = int(np.ceil(W_origin / 4))
    # resize image to fit CAM
    img = Image.fromarray(img).resize((W, H), Image.LANCZOS)
    img = np.array(img)
    """ generate pseudo label and idx for train,test """
    # [20, H, W]
    cams = np.zeros((args.num_class - 1, H, W)) 
    dict_np = np.load(os.path.join(predict_root, img_name + '.npy'),
                      allow_pickle=True).item()

    cam_label = np.zeros(20)
    #for key, cam in dict_np.items():
    #    cam_label[key] = 1
    #    cams[key] = cam
    for i in range(dict_np['cam'].shape[0]):
        cam_label[dict_np['keys'][i]] = 1
        cams[dict_np['keys'][i]] = dict_np['cam'][i]
    cam_label = cam_label.astype(np.uint8)

    # [C, H_up, W_up]
    seg_label, seg_score = compute_seg_label(ori_img=img,
                                             cam_label=cam_label,
                                             norm_cam=cams, use_crf=True,
                                             confident_region=confident_region)
    
    img_dn = Image.fromarray(img).resize((W, H), Image.LANCZOS)
    img_dn = np.asarray(img_dn)

    seg_score_dn = seg_score

    seg_label = seg_label.astype(np.uint8)

    seg_label_PIL_dn = seg_label
    seg_label_dn = np.asarray(seg_label_PIL_dn) 

    seg_label_torch_dn = seg_score_dn.argmax(axis=0) 

    # vote
    seg_label_dn = np.where(seg_label_torch_dn == seg_label_PIL_dn,
                            seg_label_torch_dn, 255)
    if show_infromation:
        print("seg_label.shape ", seg_label.shape)
        print("np.unique(seg_label) ", np.unique(seg_label))
        print("np.unique(seg_label_dn) ", np.unique(seg_label_dn))

    seg_label_dn_up = F.interpolate(torch.tensor(
        seg_label_dn[np.newaxis, np.newaxis, :, :], dtype=torch.float64),
                                    size=(H_origin, W_origin),
                                    mode="nearest").squeeze().numpy()

    misc_old.toimage(seg_label_dn_up, cmin=0, cmax=255, pal=colors_map,
                       mode='P').save(os.path.join(destination4visulization 
                       + '_DN_UP',"{}.png".format(img_name)))

    def save_pseudo_label(seg_score, seg_label, destination, img_name=None,
                          save_npy=True):
        """
        save label and label scores
        - img_name: str, only file name, not include extension or path
        - seg_score: numpy array, shape: [num_class,H,W] 
        - seg_label: numpy array, shape: [H,W] 
        """
        pseudo_label_dict = dict()
        img_label = load_image_label_from_xml(img_name=img_name,
                                              voc12_root=args.path4VOC_root)
        pseudo_label_dict[0] = seg_score[0]
        # VOC dataset: key range 0~20
        for key in img_label: 
            pseudo_label_dict[int(key + 1)] = seg_score[int(key + 1)]
        # save score
        if save_npy:
            destination_np = destination4logit
            if not os.path.exists(destination_np):
                os.mkdir(destination_np)
            np.save(os.path.join(destination_np, img_name), pseudo_label_dict)
        # save mask
        misc_old.toimage(seg_label, cmin=0, cmax=255, pal=colors_map,
                           mode='P').save(os.path.join(destination,
                           "{}.png".format(img_name)))

    # save downsampled label
    save_pseudo_label(seg_score=seg_score_dn,
                      seg_label=seg_label_dn,
                      img_name=img_name,
                      destination=destination4visulization + '_DN',
                      save_npy=False)
    # save upsampled label
    save_pseudo_label(seg_score=seg_score,
                      seg_label=seg_label,
                      img_name=img_name,
                      destination=destination4visulization + '_UP',
                      save_npy=False)

                  
def PPL_generate(num_cpu=1):
    """
    parallelly generate partial pseudo label
    """
    topredion_rate = args.confident_ratio

    pred_folder = args.path4CAM

    save_folder = args.partial_label_label

    save_folder_logit = args.partial_label_logit

    folder_list = ['_DN_UP', '_DN', '_UP']

    for f_ in folder_list:
        if not os.path.exists(save_folder + f_):
            os.makedirs(save_folder + f_)

    torch.multiprocessing.set_start_method('spawn')
    device = torch.device("cuda")
    img_list = load_img_name_list(args.path4train_images)
    print("type(img_list)", type(img_list))
    p = Pool(num_cpu)
    gen_partial_label_with_ratioP = partial(gen_partial_label_with_ratio,
        predict_root=pred_folder, destination4visulization=save_folder,
        destination4logit=save_folder_logit, confident_region=topredion_rate,
        device = device)
    for _ in tqdm.tqdm(p.imap_unordered(gen_partial_label_with_ratioP,
                                        img_list),total=len(img_list)):
        pass
    p.close()
    p.join()
    evaluate_dataset_IoU(predicted_folder=save_folder + "_DN_UP")


if __name__ == "__main__":
    # generate partial pseudo label for GCN
    PPL_generate()
