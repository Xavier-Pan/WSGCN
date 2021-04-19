# Weakly-Supervised Image Semantic Segmentation Using Graph Convolutional Networks (ICME 2021)

Pytorch Implementation of WSGCNI. WSGCNI is heavily based on [1] and [2]. 

Project Page: [Link](http://mapl.nctu.edu.tw/WSGCN/)

Paper (arXiv): [Link](https://arxiv.org/abs/2103.16762)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/weakly-supervised-image-semantic-segmentation/weakly-supervised-semantic-segmentation-on-1)](https://paperswithcode.com/sota/weakly-supervised-semantic-segmentation-on-1?p=weakly-supervised-image-semantic-segmentation)  
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/weakly-supervised-image-semantic-segmentation/weakly-supervised-semantic-segmentation-on)](https://paperswithcode.com/sota/weakly-supervised-semantic-segmentation-on?p=weakly-supervised-image-semantic-segmentation)


## My Environment
### Environment 1
- Operating System:
  - Arch Linux
- Memory
  - 64GB
- CUDA:
  - CUDA V9.0.176 
- GPU:
  - GTX 1070 8G
- Nvidia driver:
  - 460.67
- Python:
  - python 3.9.2
- Python package:
  - scipy, fire, torch, tensorboardX, pillow, torchvision, cython, tqdm, pydensecrf...
- Pytorch:
  - 1.8.1
### Environment 2
- Operating System:
  - Ubuntu 18.04  
- Memory
  - 256 GB
- CUDA:
  - CUDA V11.1.74 
- GPU:
  - NVIDIA Tesla V100 32GB x4
- Nvidia driver:
  - 450.80
- Python:
  - python 3.6.10
- Python package:
  - scipy, fire, torch, tensorboardX, pillow, torchvision, cython, tqdm, pydensecrf...
- Pytorch:
  - 1.8.0
 
## Download the VOC12 dataset
- [Visual Object Classes Challenge 2012 (VOC2012)](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
## Download the VOC12 augmentation dataset
- [Semantic Boundaries Dataset and Benchmark](https://www.dropbox.com/s/oeu149j8qtbs1x0/SegmentationClassAug.zip?dl=0)
## Download the pre-train model for IRN
- [IRN](https://drive.google.com/drive/u/1/folders/1_-AdHxZHR4_mGY3K_o1P1XNOHKD6Qq_A)
## Download the pre-train model for DeepLabV2
- [DeepLabV2-ImageNet](https://drive.google.com/drive/u/1/folders/1wxJTdFfkqHuPGotu6-oRVW0AxL-z_8gP)
- [DeepLabV2-MSCOCO](https://drive.google.com/drive/u/1/folders/166gMEci-fbmymBmLaKqyNuCfvubJTpHD)

## Setup data
### My directory structure
```
├── Data
│   ├── GCN4DeepLab
│   │   ├── Label
│   │   └── Logit
│   ├── IRN4GCN
│   │   ├── AFF_FEATURE
│   │   ├── AFF_MATRIX
│   │   ├── PARTIAL_PSEUDO_LABEL_DN
│   │   ├── PARTIAL_PSEUDO_LABEL_DN_UP
│   │   └── PARTIAL_PSEUDO_LABEL_UP
│   └── VOC12
│       ├── Split_List
│       └── VOC2012
│           ├── Annotations
│           ├── ImageSets
│           │   ├── Action
│           │   ├── Layout
│           │   ├── Main
│           │   └── Segmentation
│           ├── JPEGImages
│           ├── SegmentationClass
│           ├── SegmentationClassAug
│           └── SegmentationObject
├── GCN
│   └── runs
└── IRN
    ├── misc
    ├── net
    ├── result
    │   ├── cam
    │   ├── ins_seg
    │   ├── ir_label
    │   └── sem_seg
    ├── sess
    ├── step
    └── voc12
        └── Split_List
```

## StageI (See train.sh for more details)
```
./train.sh
cd GCN/
python CRF.py
```
## StageII (Please refer to another github)
- [DeepLabV2-ImageNet](https://github.com/johnnylu305/deeplab-imagenet-pytorch)
- [DeepLabV2-MSCOCO](https://github.com/kazuto1011/deeplab-pytorch)

## Evaluation (See eval.py for more details)
```
python eval.py
```

## Performance
Note that you may meet the performance fluctuation, which is about 0.5%, in these simplified codes for ordinary machines. This is because of the seed in train.py and -l in train.sh. Specifically, we set a seed for the train.py instead of resetting it for each GCN. For example, the performance of StageI is around 67.7% with -l 1464 in train.sh. In addition, the performance of StageII depends on StageI and the performance fluctuation is around 0.5%.
- Performance of StageI


| set      | CRF      | mIoU    |
| :---:    | :---:    |  :---:  |
| train    |X         | 66.7%   |
| train    |O         | 68.0%   |

- Performance of StageII


| set      | pre-train      | mIoU    |
| :---:    | :---:          |  :---:  |
| val     |ImageNet         | 66.7%   |
| val     |MSCOCO           | 68.7%   |
| test    |ImageNet         | 68.8%   |
| test    |MSCOCO           | 69.3%   |

## Citation
If you find the code useful, please consider citing the paper.
```
@InProceedings{pan2021all,
author = {Shun-Yi Pan, Cheng-You Lu, Shih-Po Lee, and Wen-Hsiao Pen},
title = {Weakly-Supervised Image Semantic Segmentation Using Graph Convolutional Networks},
booktitle = {IEEE International Conference on Multimedia and Expo (ICME)},
year = {2021}
}
```

## Reference
- [[1] Ahn, Jiwoon and Cho, Sunghyun and Kwak, Suha. Weakly Supervised Learning of Instance Segmentation with Inter-pixel Relations. CVPR 2019](https://github.com/jiwoon-ahn/irn)
- [[2] Kipf, Thomas N and Welling, Max. Semi-Supervised Classification with Graph Convolutional Networks. ICLR2017](https://github.com/tkipf/pygcn)



