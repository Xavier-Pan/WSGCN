#!/bin/bash

# train for Pascal VOC12
# train_aug for Pascal VOC12 augmentation dataset
TRAIN_LIST=train
TRAIN_LIST_FILE=${TRAIN_LIST}.txt
SPLIT_DIR=Split_List
SPLIT_NAME=${SPLIT_DIR}/${TRAIN_LIST}_

cd ./Data/VOC12/
rm ${SPLIT_DIR}/*
# replace 200 with larger number if you have extra storage capacity
# the number should equal to the following "split command"
split -l 200 --suffix-length=4 --numeric-suffixes ${TRAIN_LIST_FILE}  ${SPLIT_NAME}



cd ../../IRN/voc12/
rm ${SPLIT_DIR}/*
# replace 200 with larger number if you have extra storage capacity
# the number should equal to the above "split command"
split -l 200 --suffix-length=4 --numeric-suffixes ${TRAIN_LIST_FILE}  ${SPLIT_NAME}
cd ../
for sub in voc12/${SPLIT_DIR}/*
do
    rm ../Data/IRN4GCN/AFF_MATRIX/* ../Data/IRN4GCN/AFF_FEATURE/* 
    echo Current: ${sub}
    python run_sample_for_gcn.py --train_list=${sub} --infer_list=${sub} --voc12_root ../Data/VOC12/VOC2012
    cd ../GCN/
    python make_dataset.py --train_list=../Data/VOC12/${SPLIT_DIR}/${sub##*/}
    python train.py --train_list=../Data/VOC12/${SPLIT_DIR}/${sub##*/} 
    cd ../IRN/
done
