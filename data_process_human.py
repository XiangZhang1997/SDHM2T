import os
import argparse
import pickle
import cv2
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from PIL import Image
from yaml import safe_load
from torchvision.transforms import Grayscale, Normalize, ToTensor
from utils.helpers import dir_exists, remove_files
import albumentations as A
import torchvision

def data_process(data_path, name, mode):
    save_path = os.path.join(data_path, f"{mode}_pro")
    dir_exists(save_path)
    remove_files(save_path)
    if name == "DRIVE":
        gt_path = os.path.join(data_path, "2nd_manual")
        file_list = list(sorted(os.listdir(gt_path)))
    elif name == "CHASEDB1":
        gt_path = os.path.join(data_path, "2nd_label")
        file_list = list(sorted(os.listdir(gt_path)))
    elif name == "STARE":
        # img_path = os.path.join(data_path, "images")
        gt_path = os.path.join(data_path, "snd_label_vk")
        file_list = list(sorted(os.listdir(gt_path)))

    gt_list = []
    for i, file in enumerate(file_list):
        if name == "DRIVE":
            gt = Image.open(os.path.join(gt_path, file[0:2] + "_manual2.gif"))
            gt_list.append(ToTensor()(gt))
        elif name == "CHASEDB1":
            gt = Image.open(os.path.join(gt_path, file[0:9] + '_2ndHO.png'))
            gt_list.append(ToTensor()(gt))
        elif name == "STARE":
            # img = Image.open(os.path.join(img_path, file))
            gt = Image.open(os.path.join(gt_path, file[0:6] + '.vk.ppm'))
            # img_list.append(ToTensor()(img))
            gt_list.append(ToTensor()(gt))

    # img_list_save = img_list
    # save_each_image(img_list_save, save_path, "img", name)
    save_each_image(gt_list, save_path, "gt2", name)

def save_each_image(imgs_list, path, type, name):
    for i, sub in enumerate(imgs_list):
        with open(file=os.path.join(path, f'{type}_{i}.pkl'), mode='wb') as file:
            pickle.dump(np.array(sub), file)
            print(f'save {name} {type} : {type}_{i}.pkl')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset_path', default="/home/s1/ZX/job/Vessel/datasets/CHASEDB1", type=str,
                        help='the path of dataset',required=True)
    parser.add_argument('-dn', '--dataset_name', default="CHASEDB1", type=str,
                        help='the name of dataset',choices=['DRIVE','CHASEDB1','STARE','IOSTAR','RC_SLO'],required=True) #[(16best),11,7]
    args = parser.parse_args()
    
    with open('/home/s1/ZX/job/Vessel/config.yaml', encoding='utf-8') as file:
        CFG = safe_load(file)  # 为列表类型

    data_process(args.dataset_path, args.dataset_name,mode="test")