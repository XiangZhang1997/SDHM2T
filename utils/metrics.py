import numpy as np
import torch
import cv2
from sklearn.metrics import roc_auc_score, roc_curve

from math import sqrt

class AverageMeter(object):
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = np.multiply(val, weight)
        self.count = weight
        self.initialized = True

    def add(self, val, weight):
        self.val = val
        self.sum = np.add(self.sum, np.multiply(val, weight))
        self.count = self.count + weight
        self.avg = self.sum / self.count

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    @property
    def value(self):
        return np.round(self.val, 4)

    @property
    def average(self):
        return np.round(self.avg, 4)

def get_metrics(predict, target, threshold=None, mode=None,eps = 1e-10):
    n = 4

    if torch.is_tensor(predict):
        if mode is True:
            predict = torch.sigmoid(predict).cpu().detach().numpy().flatten()
        else:
            predict = predict.cpu().detach().numpy().flatten()
        # print(predict)
    else:
        if mode is True:
            predict = torch.sigmoid(predict).flatten()
        else:
            predict = predict.flatten()

    predict_b = np.where(predict >= threshold, 1, 0)
    # print(predict_b)
    if torch.is_tensor(target):
        target = target.cpu().detach().numpy().flatten()
    else:
        target = target.flatten()

    tp = (predict_b * target).sum()
    tn = ((1 - predict_b) * (1 - target)).sum()
    fp = ((1 - target) * predict_b).sum()
    fn = ((1 - predict_b) * target).sum()

    acc = (tp + tn) / (tp + fp + fn + tn)
    pre = tp / (tp + fp)
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    iou = (tp + eps)/ (tp + fp + fn + eps)
    f1 = (2 * tp  + eps )/ (2 * tp + fp + fn + eps)

    numerator = (tp * tn) - (fp * fn) 
    denominator = sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) 
    mcc = numerator/denominator

    auc = roc_auc_score(target, predict)

    return {
            "AUC": np.round(auc, n),
            "F1": np.round(f1, n),
            "Acc": np.round(acc, n),
            "Sen": np.round(sen, n),
            "Spe": np.round(spe, n),
            "MCC": np.round(mcc, n),
            "IOU": np.round(iou, n),
        }

def count_connect_component(predict, target, threshold=None, connectivity=8):
    if threshold != None:
        predict = torch.sigmoid(predict).cpu().detach().numpy()
        # predict = predict.cpu().detach().numpy()
        predict = np.where(predict >= threshold, 1, 0)
    if torch.is_tensor(target):
        target = target.cpu().detach().numpy()
    pre_n, _, _, _ = cv2.connectedComponentsWithStats(np.asarray(
        predict, dtype=np.uint8)*255, connectivity=connectivity)
    gt_n, _, _, _ = cv2.connectedComponentsWithStats(np.asarray(
        target, dtype=np.uint8)*255, connectivity=connectivity)
    return pre_n/gt_n
