import datetime
import argparse
import torch
from bunch import Bunch
from yaml import safe_load
from torch.utils.data import DataLoader
import os
import cv2
import numpy as np
import sys

from dataset import vessel_dataset,vessel_patch_dataset
from tester import Tester
from utils import losses
from utils.helpers import get_instance
from utils.helpers import dir_exists, remove_files, double_threshold_iteration
from models import stripNet

models = stripNet

def main(data_path, weight_path, CFG, show,mode=None):
    checkpoint = torch.load(weight_path)
    CFG_ck = checkpoint['config']
    model = get_instance(models, 'model', CFG)
    loss = get_instance(losses, 'loss', CFG_ck)
    test_dataset = vessel_dataset(data_path, mode="test")
    test_loader = DataLoader(test_dataset, 1,shuffle=False,  num_workers=16, pin_memory=True)
    test = Tester(model, loss, CFG, checkpoint, test_loader, data_path, show, mode="pre") # do not use checkpoint
    test.test()

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("-dp", "--dataset_path", default="/home/s1/ZX/job/Vessel/datasets/CHASEDB1", type=str,
                        help="the path of dataset")
    parser.add_argument("-wp", "--wetght_path", default="/home/s1/ZX/job/Vessel/"
                                                        "pretrained_weights/CHASEDB1/SAM2T/"
                                                        "checkpoint-epoch20.pth", type=str,
                        help='the path of wetght.pt')
    parser.add_argument("--show", help="save predict image as 0.5 threshold",
                        required=False, default=True, action="store_true")

    args = parser.parse_args()

    with open("/home/s1/ZX/job/Vessel/config.yaml", encoding="utf-8") as file:
        CFG = Bunch(safe_load(file))

    mode = "pre"
    main(args.dataset_path, args.wetght_path,CFG, args.show, mode)
    print(format(datetime.datetime.now(), "%Y-%m-%d %H:%M:%S")) 