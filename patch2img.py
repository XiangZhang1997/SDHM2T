import torch  
import numpy as np  
from scipy.signal import convolve2d  
from PIL import ImageDraw, Image

import re
def tryint(s):                  
    try:
        return int(s)
    except ValueError:
        return s
def str2int(v_str):            
    return [tryint(sub_str) for sub_str in re.split('([0-9]+)', v_str)]
def sort_humanly(v_list): 
    return sorted(v_list, key=str2int)

import pickle
import cv2
import random
import os
from PIL import Image
import pickle

saved_pre_patch_path = "/home/s1/ZX/job/Vessel/datasets/DRIVE/save_pre_patch_dir/"
pre_path = ""

def rot90_180(image):
    rotated_img = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    rotated_img = cv2.flip(rotated_img, 0)
    return rotated_img

def merge_patches(patches, volume_size, overlap_size):

    width, height = volume_size
    patch_width, patch_height = patches.shape[2],patches.shape[2]
    overlap_width, overlap_height = overlap_size
    num_patches_x = (width - patch_width) // (patch_width - overlap_width) + 1
    num_patches_y = (height - patch_height) // (patch_height - overlap_height) + 1
    print('merge:', num_patches_x, num_patches_y)

    merged_volume = np.zeros(volume_size)
    weight_volume = np.zeros(volume_size) 
    idx = 0

    for x in range(num_patches_x):
        for y in range(num_patches_y):
            x_start = x * (patch_width - overlap_width)
            y_start = y * (patch_height - overlap_height)
            merged_volume[x_start:x_start+patch_width, y_start:y_start+patch_height] += patches[idx]
            weight_volume[x_start:x_start+patch_width, y_start:y_start+patch_height] += 1
            idx += 1
    for i in range(weight_volume.shape[0]):
        for j in range(weight_volume.shape[1]):
            if weight_volume[i][j] == 0.0 or weight_volume[i][j] == 0:
                weight_volume[i][j] = 1e-10

    merged_volume /= weight_volume 

    merged_volume = rot90_180(merged_volume)
    return merged_volume

def p2i(patch_L,w=565,h=584,overlap_w=0,overlap_h=0):

    patch_L = np.array(patch_L)
    newimg = merge_patches(patch_L,(w,h),(overlap_w, overlap_h))

    return newimg

if __name__ == '__main__':
    for i in range(20):
        pre = p2i_(saved_pre_patch_path,i,w=565,h=584,overlap_w=0,overlap_h=0) # change

        for j in range(pre.shape[0]):
            for k in range(pre.shape[1]):

                if pre[j][k]>0:
                    print("*******")
                    print(pre[j][k])