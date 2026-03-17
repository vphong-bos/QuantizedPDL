import argparse
import glob
import os
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from model.pdl import (
    DEEPLAB_V3_PLUS,
    PANOPTIC_DEEPLAB,
    PytorchPanopticDeepLab,
    build_model
)

from evaluation.eval_dataset import EvalDataset, eval_collate, build_eval_loader


def get_semantic_logits(model, x, model_category_const):
    with torch.no_grad():
        outputs = model(x)

    if model_category_const == DEEPLAB_V3_PLUS:
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        return outputs

    # PANOPTIC_DEEPLAB
    if isinstance(outputs, (tuple, list)):
        return outputs[0]

    raise TypeError(f"Unexpected model output type: {type(outputs)}")


def update_confusion_matrix(conf_mat, pred, target, num_classes=19, ignore_index=255):
    valid = target != ignore_index
    pred = pred[valid]
    target = target[valid]

    if pred.numel() == 0:
        return conf_mat

    inds = num_classes * target + pred
    conf_mat += torch.bincount(inds, minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return conf_mat


def compute_miou_from_confmat(conf_mat):
    conf_mat = conf_mat.float()
    tp = torch.diag(conf_mat)
    pos_gt = conf_mat.sum(dim=1)
    pos_pred = conf_mat.sum(dim=0)
    union = pos_gt + pos_pred - tp

    iou = tp / union.clamp(min=1)
    valid = union > 0
    miou = iou[valid].mean().item() * 100.0

    return {
        "mIoU": miou,
        "IoU_per_class": (iou * 100.0).cpu().numpy(),
    }


def evaluate_model(model, model_category_const, loader, device, max_samples=-1):
    model.eval()
    conf_mat = torch.zeros((19, 19), dtype=torch.int64, device=device)

    processed = 0
    for batch in loader:
        for sample in batch:
            image = sample["image"].unsqueeze(0).to(device=device, dtype=torch.float32)
            label = sample["label"].to(device=device)
            orig_h, orig_w = sample["orig_size"]

            logits = get_semantic_logits(model, image, model_category_const)
            logits = F.interpolate(
                logits,
                size=(orig_h, orig_w),
                mode="bilinear",
                align_corners=False,
            )

            pred = logits.argmax(dim=1).squeeze(0)
            conf_mat = update_confusion_matrix(conf_mat, pred, label)

            processed += 1
            if processed % 50 == 0:
                print(f"Processed {processed} images")

            if max_samples > 0 and processed >= max_samples:
                return compute_miou_from_confmat(conf_mat)

    return compute_miou_from_confmat(conf_mat)
