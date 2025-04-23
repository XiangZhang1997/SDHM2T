import os
import time
from datetime import datetime
import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torchvision.transforms.functional as TF
from loguru import logger
from torch.utils import tensorboard
from tqdm import tqdm
from utils.helpers import dir_exists, get_instance, remove_files, double_threshold_iteration
from utils.metrics import AverageMeter, get_metrics, get_metrics, count_connect_component
import ttach as tta
import sys
from torchsummary import summary
torch.autograd.set_detect_anomaly(True)

class Trainer:
    def __init__(self, model, CFG=None, loss=None, train_loader=None, val_loader=None):
        self.CFG = CFG
        if self.CFG.amp is True:
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
        self.loss = loss
        self.model = nn.DataParallel(model.cuda())
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = get_instance(torch.optim, "optimizer", CFG, self.model.parameters())
        self.lr_scheduler = get_instance(torch.optim.lr_scheduler, "lr_scheduler", CFG, self.optimizer)
        start_time = datetime.now().strftime('%y%m%d%H%M%S')
        self.checkpoint_dir =  "/home/s1/ZX/job/Vessel/pretrained_weights/" + \
                               self.CFG['dataname']["type"] + "/" + self.CFG['model']["type"] + "/"
        self.writer = tensorboard.SummaryWriter(self.checkpoint_dir)
        dir_exists(self.checkpoint_dir)
        cudnn.benchmark = True

    def train(self):
        for epoch in range(1, self.CFG.epochs + 1):
            self._train_epoch(epoch)
            if self.val_loader is not None and epoch % self.CFG.val_per_epochs == 0:
                results = self._valid_epoch(epoch)
                logger.info(f'## Info for epoch {epoch} ## ')
                for k, v in results.items():
                    logger.info(f'{str(k):15s}: {v}')
            # if epoch % self.CFG.save_period == 0: # save_period=1 ---save each epoch !!!!!!!!!!!
            #     self._save_checkpoint(epoch)
            if epoch == self.CFG.epochs: # ---save final epoch only !!!!!!!!!!!!!!!
                self._save_checkpoint(epoch)

    # CS2-Netadjust learning rate (poly)
    def adjust_lr(optimizer, base_lr, iter, max_iter, power=0.9):
        lr = base_lr * (1 - float(iter) / max_iter) ** power
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def _train_epoch(self, epoch):
        self.model.train()
        # summary(self.model, input_size=(1, self.CFG.H, self.CFG.W),batch_size=self.CFG.batch_size)
        wrt_mode = 'train'
        self._reset_metrics()
        tbar = tqdm(self.train_loader, ncols=150)
        # tbar = tqdm(self.train_loader)
        tic = time.time()
        for img, gt in tbar:
            self.data_time.update(time.time() - tic)
            img = img.cuda(non_blocking=True)
            gt = gt.cuda(non_blocking=True)
            self.optimizer.zero_grad()
            if self.CFG.amp is True: #混合精度
                with torch.cuda.amp.autocast(enabled=True):
                    # single output
                    pre = self.model(img)
                    loss = self.loss(pre, gt)

                    # multi outputs
                    # final_1, final_2, final_3, pre = self.model(img)
                    # loss1 = self.loss(final_1, gt)
                    # loss2 = self.loss(final_2, gt)
                    # loss3 = self.loss(final_3, gt)
                    # loss4 = self.loss(pre, gt)
                    # loss = 0.25 * loss4 + 0.25 * loss3 + 0.25 * loss2 + 0.25 * loss1

                    # final_1, final_2, final_3, final_4, pre = self.model(img)
                    # loss1 = self.loss(final_1, gt)
                    # loss2 = self.loss(final_2, gt)
                    # loss3 = self.loss(final_3, gt)
                    # loss4 = self.loss(final_4, gt)
                    # loss5 = self.loss(pre, gt)
                    # loss = 0.2 * loss5 + 0.2 * loss4 + 0.2 * loss3 + 0.2 * loss2 + 0.2 * loss1


                    # final_1, final_2, pre = self.model(img)
                    # loss1 = self.loss(final_1, gt)
                    # loss2 = self.loss(final_2, gt)
                    # loss3 = self.loss(pre, gt)
                    # loss = 0.5 * loss3 + 0.3 * loss2 + 0.2 * loss1

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
#                 y = edge_conv2d(img)
#                 pre = self.model(img,y)  # y is x(img) after Sobel

                # single output
                pre = self.model(img)
                loss = self.loss(pre, gt)

                # multi outputs
                # u2
                # final_1, final_2, final_3, pre = self.model(img)
                # loss1 = self.loss(final_1, gt)
                # loss2 = self.loss(final_2, gt)
                # loss3 = self.loss(final_3, gt)
                # loss4 = self.loss(pre, gt)
                # loss = 0.25 * loss4 + 0.25 * loss3 + 0.25 * loss2 + 0.25 * loss1

                # u3
                # final_1, final_2, final_3, final_4, pre = self.model(img)
                # loss1 = self.loss(final_1, gt)
                # loss2 = self.loss(final_2, gt)
                # loss3 = self.loss(final_3, gt)
                # loss4 = self.loss(final_4, gt)
                # loss5 = self.loss(pre, gt)
                # loss = 0.2 * loss5 + 0.2 * loss4 + 0.2 * loss3 + 0.2 * loss2 + 0.2 * loss1

                # transfuse
                # final_1, final_2, pre = self.model(img)
                # loss1 = self.loss(final_1, gt)
                # loss2 = self.loss(final_2, gt)
                # loss3 = self.loss(pre, gt)
                # loss = 0.5 * loss3 + 0.3 * loss2 + 0.2 * loss1
                # loss = (1/3) * loss3 + (1/3) * loss2 + (1/3)* loss1

                loss.backward()
                self.optimizer.step()
            self.total_loss.update(loss.item())
            self.batch_time.update(time.time() - tic)

            self._metrics_update(
                *get_metrics(pre, gt, threshold=self.CFG.threshold, mode=self.CFG.Train).values())
            tbar.set_description(
                'TRAIN ({}/{}) Loss: {:.4f} | AUC {:.4f} F1 {:.4f} Acc {:.4f}'
                '  Sen {:.4f} Spe {:.4f} IOU {:.4f} MCC {:.4f}'.format(
                    epoch,self.CFG.epochs, self.total_loss.average, *self._metrics_ave().values()))
                    # self.batch_time.average, self.data_time.average))
            tic = time.time()
        self.writer.add_scalar(
            f'{wrt_mode}/loss', self.total_loss.average, epoch)
        for k, v in list(self._metrics_ave().items())[:-1]:
            self.writer.add_scalar(f'{wrt_mode}/{k}', v, epoch)
        for i, opt_group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar(
                f'{wrt_mode}/Learning_rate_{i}', opt_group['lr'], epoch)
        self.lr_scheduler.step()
        # self.adjust_lr(self.optimizer,self.CFG['optimizer']["args"]["lr"],epoch,self.CFG['epochs'])

    def _valid_epoch(self, epoch):
        logger.info('\n###### EVALUATION ######')
        self.model.eval()
        wrt_mode = 'val'
        self._reset_metrics()
        tbar = tqdm(self.val_loader, ncols=150)
        # tbar = tqdm(self.val_loader)
        with torch.no_grad():
            for img, gt in tbar:
                img = img.cuda(non_blocking=True)
#                 y = edge_conv2d(img).cuda(non_blocking=True)
                gt = gt.cuda(non_blocking=True)
                if self.CFG.amp is True: # 混合精度
                    with torch.cuda.amp.autocast(enabled=True):
#                         predict = self.model(img,y) # self.model(img, y) # y is x(img) after Sobel
                        predict = self.model(img)
                        loss = self.loss(predict, gt)
                else:
                    # single output
                    predict = self.model(img)
                    # predict = self.model(img,y) # self.model(img, y) # y is x(img) after Sobel
                    loss = self.loss(predict, gt)

                    # multi outputs
                    # _, _, _, predict = self.model(img)
                    # _, _, _, _, predict = self.model(img)
                    # _, _, predict = self.model(img)
                    # loss = self.loss(predict, gt)

                self.total_loss.update(loss.item())
                self._metrics_update(
                    *get_metrics(predict, gt, threshold=self.CFG.threshold).values())
                tbar.set_description(
                    'EVAL ({}/{})  | Loss: {:.4f} | AUC {:.4f} F1 {:.4f} Acc {:.4f} Sen {:.4f} Spe {:.4f} MCC {:.4f} IOU {:.4f}|'.format(
                        epoch,self.CFG.epochs, self.total_loss.average, *self._metrics_ave().values()))
                self.writer.add_scalar(
                    f'{wrt_mode}/loss', self.total_loss.average, epoch)

        self.writer.add_scalar(
            f'{wrt_mode}/loss', self.total_loss.average, epoch)
        for k, v in list(self._metrics_ave().items())[:-1]:
            self.writer.add_scalar(f'{wrt_mode}/{k}', v, epoch)
        log = {
            'val_loss': self.total_loss.average,
            **self._metrics_ave()
        }
        return log

    def _save_checkpoint(self, epoch):
        state = {
            'arch': type(self.model).__name__,
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'config': self.CFG
        }
        filename = os.path.join(self.checkpoint_dir,f'checkpoint-epoch{epoch}.pth')
        logger.info(f'Saving a checkpoint: {filename} ...')
        torch.save(state, filename)
        return filename

    def _reset_metrics(self):
        self.batch_time = AverageMeter()
        self.data_time = AverageMeter()
        self.total_loss = AverageMeter()
        self.auc = AverageMeter()
        self.f1 = AverageMeter()
        self.acc = AverageMeter()
        self.sen = AverageMeter()
        self.spe = AverageMeter()
        self.mcc = AverageMeter()
        self.iou = AverageMeter()
        self.CCC = AverageMeter()
        
    def _metrics_update(self, auc,f1, acc, sen, spe, mcc, iou):
        self.auc.update(auc)
        self.f1.update(f1)
        self.acc.update(acc)
        self.sen.update(sen)
        self.spe.update(spe)
        self.mcc.update(mcc)
        self.iou.update(iou)

    def _metrics_ave(self):

        return {
            "AUC": self.auc.average,
            "F1": self.f1.average,
            "Acc": self.acc.average,
            "Sen": self.sen.average,
            "Spe": self.spe.average,
            "MCC": self.mcc.average,
            "IOU": self.iou.average,
        }
