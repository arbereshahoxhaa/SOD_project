import torch

def iou_score(pred, target):
    pred = pred > 0.5
    target = target > 0.5

    intersection = (pred & target).float().sum()
    union = (pred | target).float().sum()

    return intersection / (union + 1e-8)