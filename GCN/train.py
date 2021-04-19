from __future__ import division
from __future__ import print_function

import time
import numpy as np
import os
import scipy
import scipy.misc
import misc_old
import torch
import torch.nn.functional as F
import torch.optim as optim
import datetime
from tensorboardX import SummaryWriter
from PIL import Image
from cv2 import imread, imwrite

from config import opt as args
from models import GCN
from dataset import graph_voc
from utils import IOUMetric, colors_map, evaluate_dataset_IoU
from utils import load_image_label_from_xml, SEG_ID_TO_NAME, load_img_name_list
from utils import HLoss

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def postprocess_image_save(model_output, img_name,
                           save_prediction_np=False):
    """
    1.upsample prediction scores
    2.save prediction scores (option)
    3.save prediction mask
    """
    # load image as nd_array
    img = imread(os.path.join(args.path4Image, img_name + ".jpg"))
    H_original, W_original, C = img.shape
    H = int(np.ceil(H_original / args.output_rate))
    W= int(np.ceil(W_original/ args.output_rate))

    # [H*W,num_class] -> [num_class,H,W]
    model_output = model_output.reshape(H, W,
                                        model_output.size()[-1]).permute(
                                            2, 0, 1)

    # [C,H,W] -> [1,C,H,W]
    model_output = model_output.unsqueeze(dim=0) 
    # 1.upsample the predicted mask
    upsampling = torch.nn.Upsample(size=(H_original, W_original),
                                   mode='bilinear',
                                   align_corners=True)
    # [C,H,W] -> [C, H_original, W_original] 
    up_predict = upsampling(model_output).squeeze(dim=0)

    # [C, H_original, W_original] -> [1, H_original, W_original]
    up_predict_mask = torch.argmax(up_predict, dim=0)

    # 2.save the prediction score
    if save_prediction_np:
        path = args.path4GCN_logit
        if not os.path.exists(path):
            os.makedirs(path)
            print("GCN prediction save path:", path)
        # p = exp{log(p)}
        up_predict_np = torch.exp(up_predict.clone()).cpu().numpy()
        img_label = load_image_label_from_xml(img_name=img_name,
                                              voc12_root=args.path4VOC_root)
        predict_dict = dict()
        predict_dict[0] = up_predict_np[0]
        for idx, cls_ in enumerate(img_label):
            if int(cls_) > 0:
                print("key:{} ID:{}".format(idx + 1, SEG_ID_TO_NAME[idx + 1]))
                predict_dict[idx + 1] = up_predict_np[idx + 1]
        np.save(os.path.join(path, img_name + ".npy"), predict_dict)

    # 3.save the prediction as label
    path4save = args.path4GCN_label
    if not os.path.isdir(path4save):
        os.makedirs(path4save)
    misc_old.toimage(up_predict_mask.cpu().numpy(), cmin=0, cmax=255, 
                       pal=colors_map, mode='P').save(
                       os.path.join(path4save, img_name + '.png'))
    print("Postprocessing image:{} save in {}".format(img_name, path4save))


def test(model, features, labels, adj, idx_train, img_name, epoch, t4epoch):
    """
    1.evaluate loss, accuracy, and IoU
    2.save IoU and accuracy in "evaluation4image.txt"
    """
    model.eval()
    output = model(features, adj).detach()
    predictions = torch.argmax(output, dim=1).cpu().numpy()
    mask_gt = Image.open(os.path.join(args.path4Class, img_name + '.png'))
    mask_gt = np.asarray(mask_gt)
    # 1.evaluate loss
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])

    # 1.evaluate accuracy and IoU
    IoU_one_image = IOUMetric(args.num_class)
    IoU_one_image.add_batch(predictions.cpu().numpy(), mask_gt)
    acc, acc_cls, iu, mean_iu_tensor, fwavacc = IoU_one_image.evaluate()
    # show information
    print("[{:03d}]=== Information:\n".format(epoch + 1),
          'mean_IoU: {:>8.5f}'.format(mean_iu_tensor.item()),
          'acc: {:>11.5f}'.format(acc),
          'loss_train: {:<8.4f}'.format(loss_train.item()),
          'time: {:<8.4f}s'.format(time.time() - t4epoch))

    # 2.save information
    print("save accuracy and IoU:" + img_name + ' predict')
    with open("evaluation4image.txt", 'a') as f:
        f.write(img_name + "\t")
        f.write("IoU" + str(mean_iu_tensor.item()) + "\t")
        f.write("Acc:" + str(acc) + "\n")


def evaluate_IoU(model=None, features=None, adj=None, img_name=None, 
                 img_idx=0, writer=None, IoU=None, save_prediction_np=False):
    """
    evaluate IoU of batch images
    mask_predict: numpy
    """
    model.eval()
    predictions_score = model(features, adj).detach()

    # upsample prediction and save predicted mask
    postprocess_image_save(img_name=img_name, model_output=predictions_score,
                           save_prediction_np=save_prediction_np)

    # load ground truth
    mask_gt = Image.open(os.path.join(args.path4VOC_class_aug, img_name + '.png'))
    mask_gt = np.asarray(mask_gt)

    # load prediction without CRF
    mask_predit = Image.open(
        os.path.join(args.path4GCN_label, img_name + '.png'))
    mask_predit = np.asarray(mask_predit)

    # evaluate IoU and accuracy
    IoU.add_batch(mask_predit, mask_gt)
    acc, acc_cls, iu, mean_iu_tensor, fwavacc = IoU.evaluate()
    if args.use_TB and writer:
        writer.add_scalar("IoU/{}".format(args.process_id), 
                          mean_iu_tensor.cpu().numpy(), global_step=img_idx)
    print("Acc: {:>11.2f} IoU: {:>11.2f} %".format(acc,
        mean_iu_tensor.cpu().numpy() * 100))


def compute_lap_test(data, device, radius=1):
    """
    compute laplacian
    """
    H, W, C = data["rgbxy_t"].shape
    feat = data["rgbxy_t"].reshape(-1, C)
    A = torch.ones([H * W, H * W], dtype=torch.float64) * float("Inf")
    A_xy = A.clone()
    var_rgb = 12
    var_xy = 20

    def find_neibor(card_x, card_y, H, W, radius=2):
        """
        return index of neibors of (x,y) in list
        """
        neibors_idx = []
        for idx_x in np.arange(card_x, card_x + radius + 1):
            for idx_y in np.arange(card_y - radius, card_y + radius + 1):
                 if (0 <= idx_x < H) and (0<= idx_y < W):
                    neibors_idx.append(idx_x * W + idx_y)
        return neibors_idx

    neibors = dict()
    cur_idx = 0
    for coo_x in range(H):
        for coo_y in range(W):
            neibors = find_neibor(coo_x, coo_y, H, W, radius=radius)
            for nei in neibors:
                diff_rgb = feat[cur_idx, :3] - feat[nei, :3]
                diff_xy = feat[cur_idx, 3:] - feat[nei, 3:]
                A[cur_idx, nei] = torch.sum(torch.pow(diff_rgb, 2))
                A_xy[cur_idx, nei] = torch.sum(torch.pow(diff_xy, 2))
            cur_idx += 1

    A = torch.exp(-A / var_rgb - A_xy / var_xy)
    # A(i, j) = max(A(i, j), A(j, i))
    # A(j, i) = max(A(i, j), A(j, i)) 
    A, _ = torch.max(torch.stack([A, A.transpose(1, 0)]), dim=0)
    D = torch.diag(A.sum(dim=1))
    L_mat = D - A
    return L_mat


def gcn_train(**kwargs):
    """
    GCN training
    ---
    - the folder you need:
        - args.path4AffGraph
        - args.path4node_feat
        - path4partial_label
    - these folder would be created:
        - data/GCN4DeepLab/Label
        - data/GCN4DeepLab/Logit
    """
    t_start = time.time()
    # update config
    args.parse(**kwargs)
    device = torch.device("cuda:" + str(kwargs["GPU"]))
    print(device)

   
    # tensorboard
    if args.use_TB:
        time_now = datetime.datetime.today()
        time_now = "{}-{}-{}|{}-{}".format(time_now.year, time_now.month,
                                           time_now.day, time_now.hour,
                                           time_now.minute // 30)

        keys_ignore = ["start_index", "GPU"]
        comment_init = ''
        for k, v in kwargs.items():
            if k not in keys_ignore:
               comment_init += '|{} '.format(v)
        writer = SummaryWriter(
            logdir='runs/{}/{}'.format(time_now, comment_init))

    # initial IoUMetric object for evaluation
    IoU = IOUMetric(args.num_class)
    
    # initial dataset
    train_dataloader = graph_voc(start_idx=kwargs["start_index"],
                                 end_idx=kwargs["end_index"],
                                 device=device)

    # train a seperate GCN for each image 
    t4epoch = time.time()
    for ii, data in enumerate(train_dataloader):
        if data is None:
            continue
        img_label = load_image_label_from_xml(img_name=data["img_name"],
                                              voc12_root=args.path4VOC_root)
        img_class = [idx + 1 for idx, f in enumerate(img_label) if int(f)==1]
        num_class = np.max(img_class) + 1
        model = GCN(nfeat=data["features_t"].shape[1], nhid=args.num_hid_unit,
                    nclass=args.num_class, dropout=args.drop_rate)
        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)

        # put data into GPU
        if args.cuda:
            model.to(device)
            data["features_t"] = data["features_t"].to(device)
            data["adj_t"] = data["adj_t"].to(device)
            data["labels_t"] = data["labels_t"].to(device)
            data["label_fg_t"] = data["label_fg_t"].to(device)
            data["label_bg_t"] = data["label_bg_t"].to(device)

        t_be = time.time()

        H, W, C = data["rgbxy_t"].shape
        N = H * W
        # laplacian
        if args.use_lap:
            L_mat = compute_lap_test(data, device, radius=2).to(device) 
            print("Time for laplacian {:3.1f} s".format(time.time() - t_be))

        criterion_ent = HLoss()
        for epoch in range(args.max_epoch):
            model.train()
            optimizer.zero_grad()
            output = model(data["features_t"], data["adj_t"])

            # foreground and background loss
            loss_fg = F.nll_loss(output, data["label_fg_t"], ignore_index=255)
            loss_bg = F.nll_loss(output, data["label_bg_t"], ignore_index=255)
            loss = loss_fg + loss_bg
            if args.use_ent:
                loss_entmin = criterion_ent(output,
                                            data["labels_t"],
                                            ignore_index=255)
                loss += 10. * loss_entmin
            if args.use_lap:
                loss_lap = torch.trace(
                    torch.mm(output.transpose(1, 0),
                             torch.mm(L_mat.type_as(output), output))) / N

                gamma = 1e-2
                loss += gamma * loss_lap

            if loss is None:
                print("skip this image: ", data["img_name"])
                break

            loss_train = loss.cuda()
            loss_train.backward()
            optimizer.step()

            # save predicted mask and IoU at max epoch
            if (epoch + 1) % args.max_epoch == 0 and args.save_mask:
                t_now = time.time()
                evaluate_IoU(model=model, features=data["features_t"],
                             adj=data["adj_t"], img_name=data["img_name"],
                             img_idx=ii + 1, writer=writer, IoU=IoU,
                             save_prediction_np=True)
                print("evaluate time: {:3.1f} s".format(time.time() - t_now))
                print("[{}/{}] time: {:.1f}s\n\n".format(
                    ii + 1, len(train_dataloader), t_now - t4epoch))
                t4epoch = t_now
                print("======================================")

    if writer is not None:
        writer.close()
    print("training was Finished!")
    print("Total time elapsed: {:.0f} h {:.0f} m {:.0f} s\n".format(
        (time.time() - t_start) // 3600, (time.time() - t_start) / 60 % 60,
        (time.time() - t_start) % 60))


def train(n_split=1, process_id=1, GPU_id=0, use_lap=True):
    """
    train whole dataset by calling train()
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_id)

    time_now = datetime.datetime.today()
    time_now = "{}_{}_{}_{}h".format(time_now.year, time_now.month,
                                     time_now.day, time_now.hour)

    descript = "dataset: {}, graph: {}, feature: {}, partial label: {}".format(
        os.path.basename(args.path4Data), os.path.basename(args.path4AffGraph),
        os.path.basename(args.path4node_feat),
        os.path.basename(args.path4partial_label_label))

    print("descript ", descript)

    #args.path4GCN_label = os.path.join(args.path4GCN_label, time_now)
    #args.path4GCN_logit = os.path.join(args.path4GCN_logit, time_now)

    # Split dataset 
    len_dataset = len(load_img_name_list(args.path4train_images))
    chunk = int(np.ceil(len_dataset / n_split))
    start_idx = chunk * (int(process_id) - 1)
    end_idx = start_idx + chunk if (start_idx +
                                    chunk) < len_dataset else len_dataset

    # Train a separate GCN for each image
    gcn_train(descript=descript,
          start_index=start_idx,
          end_index=end_idx,
          GPU=GPU_id)


if __name__ == "__main__":
    train()
