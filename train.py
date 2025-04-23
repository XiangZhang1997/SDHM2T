import warnings
warnings.filterwarnings("ignore")

import datetime
import argparse
from bunch import Bunch

from loguru import logger
from yaml import safe_load
from torch.utils.data import DataLoader
import sys

from dataset import vessel_dataset
from trainer import Trainer
from utils import losses
from utils.helpers import get_instance, seed_torch

from models import stripNet


models = stripNet

def main(CFG, data_path, batch_size, with_val=False):
    seed_torch()
    if with_val:
        train_dataset = vessel_dataset(data_path, mode="training", split=0.1)
        val_dataset = vessel_dataset(data_path, mode="training", split=0.1, is_val=True)
        print(len(val_dataset),len(train_dataset))
        val_loader = DataLoader(val_dataset, batch_size, shuffle=False, num_workers=16, pin_memory=True, drop_last=False)
    else:
        train_dataset = vessel_dataset(data_path, mode="training")
    
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    logger.info('The patch number of train is %d' % len(train_dataset))

    model = get_instance(models, 'model', CFG)
    logger.info(f'\n{model}\n')
    loss = get_instance(losses, 'loss', CFG)
    trainer = Trainer(
        model=model,
        loss=loss,
        CFG=CFG,
        train_loader=train_loader,
        val_loader = val_loader if with_val else None
    )

    trainer.train()

"""
python /home/s1/ZX/job/Vessel/train.py -dp "/home/s1/ZX/job/Vessel/datasets/DRIVE" --val
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-dp', '--dataset_path', default="/home/s1/ZX/job/Vessel/datasets/DRIVE", type=str,
                        help='the path of dataset')
    parser.add_argument('-bs', '--batch_size', default=32,
                        help='batch_size for trianing and validation')
    parser.add_argument("--val", help="split training data for validation",
                        required=True, default=True, action="store_true")
    args = parser.parse_args()

    with open('/home/s1/ZX/job/Vessel/config.yaml', encoding='utf-8') as file:
        CFG = Bunch(safe_load(file))
        print(CFG)
    main(CFG, args.dataset_path, args.batch_size, args.val)
    print(format(datetime.datetime.now(), "%Y-%m-%d %H:%M:%S")) 
