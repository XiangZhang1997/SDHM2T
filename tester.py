import time
import cv2
import torch
import os
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from loguru import logger
from tqdm import tqdm
from trainer import Trainer
from utils.helpers import dir_exists, remove_files, double_threshold_iteration
from utils.metrics import AverageMeter, get_metrics, get_metrics, count_connect_component
import ttach as tta
import torch.nn.functional as F
import time

import pickle
import sys

from patch2img import p2i
from patch2img3 import p2i_3
from dataset import vessel_patch_dataset

def load_matching_weights(model, pretrained_weights):
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_weights.items() if k in model_dict and model_dict[k].size() == v.size()}
    model_dict.update(pretrained_dict)

    model.load_state_dict(model_dict)

class Tester(Trainer):
    def __init__(self, model, loss, CFG, checkpoint=None, test_loader=None, dataset_path=None, show=True, mode=None):
        # super(Trainer, self).__init__()
        self.loss = loss
        self.CFG = CFG
        self.test_loader = test_loader
        self.model = nn.DataParallel(model.cuda())
        self.dataset_path = dataset_path
        self.show = show
        self.mode = mode
        self.ck = checkpoint['state_dict']
        self.dir = self.CFG['save_dir'] + self.CFG['dataname']["type"] + "/" + self.CFG['model']["type"] + "/"+self.CFG['model']["type"]+"_pre_save_picture/"

        self.dir2 = "/home/s1/ZX/job/Vessel/datasets/"+self.CFG['dataname']["type"]+"/test_pro"

        load_matching_weights(self.model, self.ck)
        dir_exists(self.dir)
        remove_files(self.dir)
        cudnn.benchmark = True

    def adjust_brightness(self, image, beta):

        image_float = image.astype(np.float32)
        # 调整亮度
        image_bright = cv2.convertScaleAbs(image_float, alpha=1, beta=beta)
        return image_bright

    def patch_img2pre(self,test_patch_loader):
        if self.CFG.tta:
            self.model = tta.SegmentationTTAWrapper(
                self.model, tta.aliases.d4_transform(), merge_mode='mean')
        self.model.eval()
        tbar = tqdm(test_patch_loader, ncols=150)
        patch_L = []
        j_L = []
        H = []
        # start
        with torch.no_grad():
            for j, img in enumerate(tbar):
                img = img.cuda(non_blocking=True)
                # multi outputs
                # _, _, _, predict = self.model(img) # u2
                # _, _, _, _, predict = self.model(img) # u3
                # _,_,predict = self.model(img)

                # single outputs
                predict = self.model(img)
                
                predict = predict[0,0,...]
                predict = torch.sigmoid(predict).cpu().numpy()
                heatmap = np.uint8(255 * predict)


                patch_L.append(predict)
                j_L.append(j)
                H.append(heatmap)
        if self.CFG.heatmap is True:
            return patch_L,j_L,H
        else:
            return patch_L, j_L

    def test(self):
        pre_L = []
        if "DRIVE" in self.CFG['dataname']["type"]:
            self.data_len = 20
            self.w = 565
            self.h = 584
            self.overlap_w = 48
            self.overlap_h = 48
        elif "STARE" in self.CFG['dataname']["type"]:
            self.data_len = 10
            self.w = 700
            self.h = 605
            self.overlap_w = 48
            self.overlap_h = 48
        elif "CHASEDB1"  in self.CFG['dataname']["type"]:
            self.data_len = 8
            self.w = 999
            self.h = 960
            self.overlap_w = 72
            self.overlap_h = 72
        elif "RC_SLO"  in self.CFG['dataname']["type"]:
            self.data_len = 10
            self.w = 360
            self.h = 320
            self.overlap_w = 36
            self.overlap_h = 36
        elif "IOSTAR" in self.CFG['dataname']["type"]:
            self.data_len = 10
            self.w = 1024
            self.h = 1024
            self.overlap_w = 72
            self.overlap_h = 72
        elif "CHUAC" in self.CFG['dataname']["type"]:
            self.data_len = 10
            self.w = 512
            self.h = 512
            self.overlap_w = 56
            self.overlap_h = 56

        for i in range(self.data_len):
            img_patch_data_path = os.path.join(self.dataset_path,f"{self.mode}_pro",str(i))
            save_pre_patch_dir = os.path.join(self.dataset_path,f"save_{self.mode}_patch_dir")
            img_patch_dataset = vessel_patch_dataset(img_patch_data_path) # data_process
            # print(len(img_patch_dataset))
            img_patch__loader = DataLoader(img_patch_dataset, 1,shuffle=False,  num_workers=16, pin_memory=True) # 132
            if self.CFG.heatmap is True:
                patch_L,j_L,H = self.patch_img2pre(test_patch_loader=img_patch__loader)
            else:
                patch_L, j_L = self.patch_img2pre(test_patch_loader=img_patch__loader)

            #### save pre patch
            p = os.path.join(save_pre_patch_dir,str(i))
            print(p)
            dir_exists(p)
            remove_files(p)
            patch_heat_L = []
            if self.CFG.heatmap is True:
                for patch,j,heatmap_patch in zip(patch_L,j_L,H):
                    # heat
                    p2 = os.path.join(p,f"pre_b{j}.png")
                    p_heatmap = os.path.join(p, f"pre_heatmap{j}.png")

                    output_shape = (64, 64)  
                    heatmap_patch = torch.from_numpy(heatmap_patch)
                    heatmap_patch = heatmap_patch.unsqueeze(0).unsqueeze(0).float()
                    channel_mean = torch.mean(heatmap_patch, dim=1,
                                              keepdim=True)  # channel_max,_ = torch.max(feats,dim=1,keepdim=True)
                    channel_mean = F.interpolate(channel_mean, size=output_shape, mode='bilinear', align_corners=False)
                    channel_mean = channel_mean.squeeze(0).squeeze(0).cpu().detach().numpy()  
                    channel_mean = (
                            ((channel_mean - np.min(channel_mean)) / (
                                        np.max(channel_mean) - np.min(channel_mean))) * 255).astype(
                        np.uint8)
                    heatmap_patch = cv2.applyColorMap(channel_mean, cv2.COLORMAP_JET)
                    patch_heat_L.append(heatmap_patch)
                    cv2.imwrite(p_heatmap, np.uint8(heatmap_patch)) # right
                    # no heat
                    cv2.imwrite(p2, np.uint8(patch*255)) # right

                # 若路径都对，那就是h&w的顺序不对！！！！！！！！！！！！
                # pre = p2i(patch_L,w=565,h=584,overlap_w=48,overlap_h=48) ############change
                # heat
                pre_heat = p2i_3(patch_heat_L,w=self.w,h=self.h,overlap_w=self.overlap_w,overlap_h=self.overlap_h) ############change
                pre = p2i(patch_L,w=self.w,h=self.h,overlap_w=self.overlap_w,overlap_h=self.overlap_h) ############change
                # pre = p2i_(save_pre_patch_dir,i,w=565,h=584,overlap_w=48,overlap_h=48) ############change
            else:
                for patch, j in zip(patch_L, j_L):
                    # heat
                    p2 = os.path.join(p, f"pre_b{j}.png")
                    # no heat
                    cv2.imwrite(p2, np.uint8(patch * 255))  # right

                # 若路径都对，那就是h&w的顺序不对！！！！！！！！！！！！
                # pre = p2i(patch_L,w=565,h=584,overlap_w=48,overlap_h=48) ############change
                # heat
                # pre_heat = p2i_3(patch_heat_L,w=self.w,h=self.h,overlap_w=self.overlap_w,overlap_h=self.overlap_h) ############change
                pre = p2i(patch_L, w=self.w, h=self.h, overlap_w=self.overlap_w, overlap_h=self.overlap_h)  ############change
                # pre = p2i_(save_pre_patch_dir,i,w=565,h=584,overlap_w=48,overlap_h=48) ############change


            # for vis
            if self.show:
                # heat
                if self.CFG.heatmap is True:
                    predict_heat = torch.from_numpy(pre_heat.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                    predict_heat = predict_heat[0,0,...]
                    predict_heat = predict_heat.cpu().detach().numpy()
                    cv2.imwrite(
                        self.dir+f"pre_heat{i}.png", np.uint8(predict_heat))

                # no heat
                predict = torch.from_numpy(pre.astype(np.float32)).unsqueeze(0).unsqueeze(0)
                predict = predict[0,0,...]
                predict_pro = predict.cpu().detach().numpy()
                predict_b = np.where(predict.cpu().detach().numpy() >= self.CFG.threshold, 1, 0)
                cv2.imwrite(
                    self.dir+f"pre_pro{i}.png", np.uint8(predict_pro*255))
                cv2.imwrite(
                    self.dir+f"pre_b{i}.png", np.uint8(predict_b*255))

                # 未二值化 for metrics
                with open(file=os.path.join(self.dir, f'pre_{i}.pkl'), mode='wb') as file:
                    print(file)
                    pickle.dump(pre, file)
                    print(f'save pre_{i}.pkl')


