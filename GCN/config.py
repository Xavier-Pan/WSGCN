import os
import torch
import warnings
import argparse

class DefaultConfig(object):

    debug = False
    seed = 42
    cuda = torch.cuda.is_available()
    use_TB = True
    process_id = 1

    # === parameters for GCN ===
    lr = 0.01
    weight_decay = 5e-4
    num_class = 21
    max_epoch = 250
    num_hid_unit = 16
    drop_rate = .3
    use_lap = True
    use_ent = True

    # === parameter for preprocessing ===
    confident_ratio = 0.3

    # === VOC dataset ===
    path4Data = os.path.join("..", "Data")
    path4VOC_root = os.path.join(path4Data, "VOC12", "VOC2012")
    path4Image = os.path.join(path4VOC_root, "JPEGImages")
    # path4Class
    path4VOC_class_aug = os.path.join(path4VOC_root, "SegmentationClassAug")
    # image list
    path4train_images = os.path.join(path4Data, "VOC12", "train.txt")
    path4train_aug_images = os.path.join(path4Data, "VOC12", "train_aug2.txt")
    path4val_images = os.path.join(path4Data, "VOC12", "val3.txt")
    path4trainval_images = os.path.join(path4Data, "trainval4.txt")
    eval_dataset = True

    # === IRN4GCN ===
    path4IRN4GCN = os.path.join(path4Data, "IRN4GCN")
    # path4boundaryMap 
    path4CAM = os.path.join("../IRN/result/cam")
    # path4boundaryMap_logit 
    path4CAMLogit = os.path.join(path4IRN4GCN, "CAMLogit")
    path4AffGraph = os.path.join(path4IRN4GCN, "AFF_MATRIX")
    path4node_feat = os.path.join(path4IRN4GCN, "AFF_FEATURE")
    partial_label_label = os.path.join(path4IRN4GCN, "PARTIAL_PSEUDO_LABEL")
    partial_label_logit = os.path.join(path4IRN4GCN, "PARTIAL_PSEUDO_LABEL_LOGIT")
    path4partial_label_label = os.path.join(partial_label_label+"_DN")
    path4partial_label_logit = os.path.join(partial_label_logit+"_DN")

    output_rate = 4

    # === parameter for postprocessing ===
    path4GCN4DeepLab = os.path.join(path4Data, "GCN4DeepLab")
    path4GCN_logit = os.path.join(path4GCN4DeepLab, "Logit")
    path4GCN_label = os.path.join(path4GCN4DeepLab, "Label")
    path4Complete_label_label = os.path.join(path4GCN4DeepLab, "CRF_Label")
    path4Complete_label_logit = os.path.join(path4GCN4DeepLab, "CRF_Logit")
    save_prediction_np = True
    save_mask = True

    # === CRF ===
    CRF = dict()
    
    CRF["iter_max"] = 10
    CRF["pos_w"] = 3
    CRF["pos_xy_std"] = 3
    CRF["bi_w"] = 3
    CRF["bi_xy_std"] = 50
    CRF["bi_rgb_std"] = 5

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_list", default=path4train_images, type=str)
    args = parser.parse_args()
    path4train_images = args.train_list


    def parse(self, **kwargs):
        """
        update config
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                continue
                #warnings.warn("Warning: opt does not have attribute:  {}".format(k))
            else:
                setattr(self, k, v)

        opt.device = torch.device('cuda') if opt.cuda else torch.device('cpu')


opt = DefaultConfig()
