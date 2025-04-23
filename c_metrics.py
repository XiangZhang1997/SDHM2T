import pandas as pd
import openpyxl
import datetime
import argparse
from bunch import Bunch
from yaml import safe_load
from torch.utils.data import DataLoader
import time
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms.functional as TF
from loguru import logger
from tqdm import tqdm
from trainer import Trainer
from utils.helpers import dir_exists, remove_files, double_threshold_iteration
from utils.metrics import AverageMeter, get_metrics, get_metrics, count_connect_component
import ttach as tta
from dataset import img_gt_pre_dataset, human_gt_pre_dataset

class Metrics(Trainer):
    def __init__(self, CFG, test_loader, dataset_path, pre_path):
        super(Trainer, self).__init__()
        self.CFG = CFG
        self.test_loader = test_loader
        self.dataset_path = dataset_path
        self.pre_path = pre_path
        cudnn.benchmark = True

    def test(self):

        tbar = tqdm(self.test_loader, ncols=150)
        # tic = time.time()
        L = []
        with torch.no_grad():
            self._reset_metrics()
            metrics_score_D = {'AUC': 0.0, 'F1': 0.0, 'Acc': 0.0, 'Sen': 0.0, 'Spe':0.0,'MCC': 0.0, 'IOU': 0.0}
            L = []
            count = 0
            for i, (gt, pre) in enumerate(tbar):

                gt = gt[0,0,...]
                pre = pre[0,...]
                cv2.imwrite(
                    self.pre_path+f"gt{i}.png", np.uint8(gt.cpu().numpy()*255))

                # Dual-Threshold Iteration
                if self.CFG.DTI:
                    pre_DTI = double_threshold_iteration(
                        i, pre, self.CFG.threshold, self.CFG.threshold_low, True)
                    self._metrics_update(
                        *get_metrics(pre, gt, predict_b=pre_DTI).values())
                    if self.CFG.CCC:
                        self.CCC.update(count_connect_component(pre_DTI, gt))
                else:
                    self._metrics_update(
                        *get_metrics(pre, gt, threshold=self.CFG.threshold).values())
                    if self.CFG.CCC:
                        self.CCC.update(count_connect_component(
                            pre, gt, threshold=self.CFG.threshold))
                    if self.CFG.CCQ:
                        self._metrics_ccq_update(
                            *clDice(pre, gt).values())
                tbar.set_description(
                    'TEST ({}) | AUC {:.4f} F1 {:.4f} Acc {:.4f}  Sen {:.4f} Spe {:.4f} MCC {:.4f} IOU {:.4f}|'.format(
                        i, *self._metrics_ave().values()))
                tbar.set_description(
                    'TEST ({}) | cldice {:.4f} hd95 {:.4f}|'.format(
                        i, *self._metrics_ccq_ave().values()))

        LL = []
        print("*" * 100)
        for k, v in self._metrics_ave().items():
            LL.append(v)
            logger.info(f'{str(k):5s}: {v}')
        print("LL",LL)
        for k, v in self._metrics_ccq_ave().items():
            logger.info(f'{str(k):5s}: {v}')
        if self.CFG.CCC:
            logger.info(f'     CCC:  {self.CCC.average}')

def main(data_path, CFG):

    pre_path = CFG['save_dir'] + CFG['dataname']["type"] + "/" + CFG['model']["type"] + "/"+CFG['model']["type"]+"_pre_save_picture/"

    test_dataset = img_gt_pre_dataset(data_path, pre_path, mode="test")
    test_loader = DataLoader(test_dataset, 1,shuffle=False,  num_workers=0, pin_memory=True)
    test = Metrics(CFG, test_loader, data_path, pre_path) # do not use checkpoint
    test.test()

import multiprocessing as mp

if __name__ == '__main__':
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument("-dp", "--dataset_path", default="/home/s1/ZX/job/Vessel/datasets/DRIVE", type=str,
                        help="the path of dataset")

    args = parser.parse_args()

    with open("/home/s1/ZX/job/Vessel/config.yaml", encoding="utf-8") as file:
        CFG = Bunch(safe_load(file))
    main(args.dataset_path,CFG)
    print(CFG["model"]["type"]+"---"+CFG["dataname"]["type"])

    print(format(datetime.datetime.now(), "%Y-%m-%d %H:%M:%S"))