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

def data_process(data_path, name, patch_size, stride, mode, gen_pre_patch=None):
    save_path = os.path.join(data_path, f"{mode}_pro")
    dir_exists(save_path)
    remove_files(save_path)
    if name == "DRIVE":
        img_path = os.path.join(data_path, mode, "images")
        gt_path = os.path.join(data_path, mode, "1st_manual")
        file_list = list(sorted(os.listdir(img_path)))
    elif name == "CHASEDB1":
        img_path = os.path.join(data_path, mode, "images")
        gt_path = os.path.join(data_path, mode, "1st_manual")
        file_list = list(sorted(os.listdir(img_path)))
    elif name == "STARE":
        img_path = os.path.join(data_path, mode, "images")
        gt_path = os.path.join(data_path, mode, "1st_labels_ah")
        file_list = list(sorted(os.listdir(img_path)))
    elif name == "IOSTAR":
        img_path = os.path.join(data_path, mode, "image")
        gt_path = os.path.join(data_path, mode, "GT")
        file_list = list(sorted(os.listdir(img_path)))
    elif name == "RC_SLO":
        img_path = os.path.join(data_path, mode, "originalImage")
        gt_path = os.path.join(data_path, mode, "GT")
        file_list = list(sorted(os.listdir(img_path)))
    elif name == "CHUAC":
        img_path = os.path.join(data_path, "Original")
        gt_path = os.path.join(data_path, "Photoshop")
        file_list = list(sorted(os.listdir(img_path)))
    elif name == "FIVES":
        img_path = os.path.join(data_path, mode, "Original")
        gt_path = os.path.join(data_path, mode, "Ground truth")
        file_list = list(sorted(os.listdir(img_path)))
    elif name == "DCA1":
        data_path = os.path.join(data_path, "Database_134_Angiograms")
        file_list = list(sorted(os.listdir(data_path)))
    elif name == "HRF":
        img_path = os.path.join(data_path, mode, "images")
        gt_path = os.path.join(data_path, mode, "manual1")
        file_list = list(sorted(os.listdir(img_path)))
    img_list = []
    gt_list = []
    for i, file in enumerate(file_list):
        if name == "DRIVE":
            img = Image.open(os.path.join(img_path, file))
            if file[-3:] == "png":
                gt = Image.open(os.path.join(gt_path, file[0:2] + "_manual1.png"))
            else:
                gt = Image.open(os.path.join(gt_path, file[0:2] + "_manual1.gif"))
            img = Grayscale(1)(img)
            img_list.append(ToTensor()(img))
            gt_list.append(ToTensor()(gt))
        elif name == "CHASEDB1":
            if len(file) == 13:
                if mode == "training" and int(file[6:8]) <= 10:
                    img = Image.open(os.path.join(img_path, file))
                    gt = Image.open(os.path.join(gt_path, file[0:9] + '_1stHO.png'))
                    img = Grayscale(1)(img)
                    img_list.append(ToTensor()(img))
                    gt_list.append(ToTensor()(gt))
                elif mode == "test" and int(file[6:8]) > 10:
                    img = Image.open(os.path.join(img_path, file))
                    gt = Image.open(os.path.join(gt_path, file[0:9] + '_1stHO.png'))
                    img = Grayscale(1)(img)
                    img_list.append(ToTensor()(img))
                    gt_list.append(ToTensor()(gt))
        elif name == "STARE":
            if not file.endswith("gz"):
                img = Image.open(os.path.join(img_path, file))
                gt = Image.open(os.path.join(gt_path, file[0:6] + '.ah.ppm'))
                img = Grayscale(1)(img)
                img_list.append(ToTensor()(img))
                gt_list.append(ToTensor()(gt))
        elif name == "IOSTAR":
            if mode == "training" and int(file[-10:-8]) <= 32:
                img = Image.open(os.path.join(img_path, file))
                gt = Image.open(os.path.join(
                    gt_path, file[:-4] + "_GT.tif"))
                img = Grayscale(1)(img)
                img_list.append(ToTensor()(img))
                gt_list.append(ToTensor()(gt))
            elif mode == "test" and int(file[-10:-8]) > 32:
                img = Image.open(os.path.join(img_path, file))
                gt = Image.open(os.path.join(
                    gt_path, file[:-4] + "_GT.tif"))
                img = Grayscale(1)(img)
                img_list.append(ToTensor()(img))
                gt_list.append(ToTensor()(gt))
        elif name == "RC_SLO":
            img = Image.open(os.path.join(img_path, file))
            gt = Image.open(os.path.join(gt_path, file[:-4]+"_GT.tif"))
            img = Grayscale(1)(img)
            img_list.append(ToTensor()(img))
            gt_list.append(ToTensor()(gt))

    img_list = normalization(img_list)
    if mode == "training":
        img_patch = get_patch(img_list, patch_size, stride)
        gt_patch = get_patch(gt_list, patch_size, stride)
        save_patch(img_patch, save_path, "img_patch", name,mode)
        save_patch(gt_patch, save_path, "gt_patch", name,mode)
    elif mode == "test":
        img_list_save = img_list
        save_each_image(img_list_save, save_path, "img", name)
        save_each_image(gt_list, save_path, "gt", name)

        if gen_pre_patch:
            for i in range(len(img_list)):
                save_pre_path = os.path.join(data_path, f"pre_pro")
                save_test_patch_path = os.path.join(save_pre_path, str(i))
                dir_exists(save_test_patch_path)
                remove_files(save_test_patch_path)
                img_patch_list = crop_volume_channel(img_list[i],patch_size, stride)
                save_patch(img_patch_list, save_test_patch_path, "img_patch", name)
    elif mode == "test_pro":
        print(save_path)
        save_path = os.path.join(save_path)
        if name != "CHUAC":

            img_patch = extract_patches_ordered(img_list, patch_size, stride)
            gt_patch = extract_patches_ordered(gt_list, patch_size, stride)

def get_square(img_list, name):
    img_s = []
    if name == "DRIVE":
        shape = 592
        # shape = 624
    elif name == "CHASEDB1":
        shape = 1008
    _, h, w = img_list[0].shape
    pad = nn.ConstantPad2d((0, shape-w, 0, shape-h), 0)
    for i in range(len(img_list)):
        img = pad(img_list[i])
        img_s.append(img)

    return img_s

def get_patch(imgs_list, patch_size, stride):
    image_list = []
    _, h, w = imgs_list[0].shape
    pad_h = stride - (h - patch_size) % stride
    pad_w = stride - (w - patch_size) % stride
    for sub1 in imgs_list:
        image = F.pad(sub1, (0, pad_w, 0, pad_h), "constant", 0) # 加pad的img
        image = image.unfold(1, patch_size, stride).unfold(
            2, patch_size, stride).permute(1, 2, 0, 3, 4)
        image = image.contiguous().view(
            image.shape[0] * image.shape[1], image.shape[2], patch_size, patch_size)
        for sub2 in image:
            image_list.append(sub2)
    return image_list

# gen patch for pre
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from PIL import Image
def crop_volume(volume, patch_size, stride):
    volume = volume.permute(0, 2, 1)
    _, width, height = volume.shape
    volume = np.array(volume)[0,:,:]
    patch_size = to_2tuple(patch_size)
    patch_width, patch_height = patch_size
    stride = to_2tuple(stride)
    stride_width, stride_height = stride

    patches = []
    for x in range(0, width - patch_width + 1, stride_width):
        for y in range(0, height - patch_height + 1, stride_height):
            # print(x, y)
            patch = np.expand_dims(volume[x:x+patch_width, y:y+patch_height],axis=0)
            patch = torch.from_numpy(patch)
            patches.append(patch)
    return patches

def crop_volume_channel(volume, patch_size, stride):
    volume = volume.permute(0, 2, 1)
    _, width, height = volume.shape
    volume = np.array(volume)[:,:,:]
    print("volume",volume.shape)
    patch_size = to_2tuple(patch_size)
    patch_width, patch_height = patch_size
    stride = to_2tuple(stride)
    stride_width, stride_height = stride
    patches = []
    for x in range(0, width - patch_width + 1, stride_width):
        for y in range(0, height - patch_height + 1, stride_height):
            patch = volume[:,x:x+patch_width, y:y+patch_height]
            patch = torch.from_numpy(patch)
            patches.append(patch)
    return patches

def save_patch(imgs_list, path, type, name, mode=None):
    for i, sub in enumerate(imgs_list):
        if mode == "test":
            file=str(os.path.join(path, f'{type}_{ii}.png'))
            sub = np.uint8(np.array(sub.permute(1, 2, 0)))
            print(sub.shape)
            cv2.imwrite(file,sub)
        else:
            with open(file=os.path.join(path, f'{type}_{i}.pkl'), mode='wb') as file:

                print(file)
                print(sub.shape)
                pickle.dump(np.array(sub), file)

def save_each_image(imgs_list, path, type, name):
    for i, sub in enumerate(imgs_list):
        with open(file=os.path.join(path, f'{type}_{i}.pkl'), mode='wb') as file:
            pickle.dump(np.array(sub), file)
            print(f'save {name} {type} : {type}_{i}.pkl')

def normalization(imgs_list):
    imgs = torch.cat(imgs_list, dim=0)
    mean = torch.mean(imgs)
    std = torch.std(imgs)
    normal_list = []
    for i in imgs_list:
        n = Normalize([mean], [std])(i)
        n = (n - torch.min(n)) / (torch.max(n) - torch.min(n))
        normal_list.append(n)
    return normal_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset_path', default="/home/s1/ZX/job/Vessel/datasets/CHASEDB1", type=str,
                        help='the path of dataset',required=True)
    parser.add_argument('-dn', '--dataset_name', default="CHASEDB1", type=str,
                        help='the name of dataset',choices=['DRIVE','CHASEDB1','STARE','IOSTAR','RC_SLO'],required=True) 
    parser.add_argument('-ps', '--patch_size', default=64, 
                        help='the size of patch for image partition')
    parser.add_argument('-s', '--stride', default=16,
                        help='the stride of image partition')
    args = parser.parse_args()
    
    with open('/home/s1/ZX/job/Vessel/config.yaml', encoding='utf-8') as file:
        CFG = safe_load(file)  
    #
    data_process(args.dataset_path, args.dataset_name,args.patch_size, args.stride, "training")
    data_process(args.dataset_path, args.dataset_name,args.patch_size, args.stride, "test",gen_pre_patch=True)