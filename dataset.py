import os
import pickle
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip,Grayscale
from utils.helpers import Fix_RandomRotation
import albumentations as A
import numpy as np
from PIL import Image
from albumentations.pytorch import ToTensorV2
import cv2
from sort_list import sort_humanly

# training
class vessel_dataset(Dataset):
    def __init__(self, path, mode="test", is_val=False, split=None):

        self.mode = mode
        self.is_val = is_val
        self.data_path = os.path.join(path, f"{mode}_pro")
        if mode == "training":
            self.data_file = os.listdir(self.data_path)
        else:
            self.data_file = sort_humanly(os.listdir(self.data_path)) # [gt_0,gt_1...,img_0]
        self.img_file = self._select_img(self.data_file)
        if split is not None and mode == "training": 
            assert split > 0 and split < 1
            if is_val:
                self.img_file = self.img_file[:int(split*len(self.img_file))] # :0.2*len(self.img_file)
            else:
                self.img_file = self.img_file[int(split*len(self.img_file)):] # 0.2*len(self.img_file):
        self.transforms = Compose([
            RandomHorizontalFlip(p=0.5),
            RandomVerticalFlip(p=0.5),
            Fix_RandomRotation(),
        ])

    def __getitem__(self, idx):
        img_file = self.img_file[idx]
        with open(file=os.path.join(self.data_path, img_file), mode='rb') as file:
            img = torch.from_numpy(pickle.load(file)).float()
        gt_file = "gt" + img_file[3:]
        with open(file=os.path.join(self.data_path, gt_file), mode='rb') as file:
            gt = torch.from_numpy(pickle.load(file)).float()

        if self.mode == "training" and not self.is_val:
            seed = torch.seed()
            torch.manual_seed(seed)
            img = self.transforms(img)
            torch.manual_seed(seed)
            gt = self.transforms(gt)

        return img, gt

    def _select_img(self, file_list):
        img_list = []
        for file in file_list:
            if file[:3] == "img" or file[:3] == "pre":
                img_list.append(file)

        return img_list

    def __len__(self):
        return len(self.img_file)

# test
class vessel_patch_dataset(Dataset):
    def __init__(self, path):

        self.data_path = path
        self.patch_img_file = sort_humanly(os.listdir(self.data_path))
        self.patch_img_file = self._select_img(self.patch_img_file)

    def __getitem__(self, idx):

        patch_img_file = self.patch_img_file[idx]
        with open(file=os.path.join(self.data_path, patch_img_file), mode='rb') as file:
            patch_img = torch.from_numpy(pickle.load(file)).float()
        return patch_img

    def _select_img(self, file_list):
        img_list = []
        for file in file_list:
            if file[:3] == "img":
                img_list.append(file)

        return img_list

    def __len__(self):
        return len(self.patch_img_file)

# metrics
class img_gt_pre_dataset(Dataset):
    def __init__(self, path, pre_path, mode="test", is_val=False, split=None):

        self.mode = mode
        self.data_path = os.path.join(path, f"{mode}_pro")
        self.data_file = sort_humanly(os.listdir(self.data_path)) # [gt_0.pkl,gt_1.pkl...,img_0]
        self.img_file = self._select_img(self.data_file)
        self.pre_data_path = pre_path

    def __getitem__(self, idx):
        img_file = self.img_file[idx]
        gt_file = "gt" + img_file[3:]
        with open(file=os.path.join(self.data_path, gt_file), mode='rb') as file:
            gt = torch.from_numpy(pickle.load(file)).float()

        pre_file = "pre" + img_file[3:]
        with open(file=os.path.join(self.pre_data_path, pre_file), mode='rb') as file:
            pre = torch.from_numpy(pickle.load(file)).float()

        return gt, pre

    def _select_img(self, file_list):
        img_list = []
        for file in file_list:
            if file[:3] == "img":
                img_list.append(file)
        return img_list

    def __len__(self):
        return len(self.img_file)

# metrics vs human
class human_gt_pre_dataset(Dataset):
    def __init__(self, path_1st, path_2nd, mode="test", is_val=False, split=None):

        self.mode = mode
        self.data_path = os.path.join(path_1st, f"{mode}_pro")
        self.data_file = sort_humanly(os.listdir(self.data_path)) # [gt_0,gt_1...,img_0]
        self.img_file = self._select_img(self.data_file)

        self.data_path2 = os.path.join(path_2nd, f"{mode}_pro")

    def __getitem__(self, idx):
        img_file = self.img_file[idx]
        gt_file = "gt" + img_file[3:]
        with open(file=os.path.join(self.data_path, gt_file), mode='rb') as file:
            gt = torch.from_numpy(pickle.load(file)).float()

        gt2_file = "gt2" + img_file[3:]
        with open(file=os.path.join(self.data_path2, gt2_file), mode='rb') as file:
            gt2 = torch.from_numpy(pickle.load(file)).float()
            # gt2 = gt
        return gt, gt2

    def _select_img(self, file_list):
        img_list = []
        for file in file_list:
            if file[:3] == "img":
                img_list.append(file)
        return img_list

    def __len__(self):
        return len(self.img_file)