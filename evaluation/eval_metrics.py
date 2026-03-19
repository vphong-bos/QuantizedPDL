import argparse
import glob
import os
import time
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


def get_semantic_logits(model_obj, x, model_category_const):
    backend = model_obj["backend"]

    if backend == "torch":
        model = model_obj["model"]
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

    elif backend == "onnx":
        session = model_obj["session"]
        input_name = model_obj["input_name"]

        x_np = x.detach().cpu().numpy().astype(np.float32)
        outputs = session.run(None, {input_name: x_np})

        # Keep same assumption as torch path: semantic logits are first output
        logits = outputs[0]

        return torch.from_numpy(logits).to(device=x.device)

    else:
        raise ValueError(f"Unsupported backend: {backend}")


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


def evaluate_model(model_obj, model_category_const, loader, device, max_samples=-1):
    if model_obj["backend"] == "torch":
        model_obj["model"].eval()

    conf_mat = torch.zeros((19, 19), dtype=torch.int64, device=device)

    processed = 0
    total_inference_time = 0.0

    for batch in loader:
        for sample in batch:
            image = sample["image"].unsqueeze(0).to(device=device, dtype=torch.float32)
            label = sample["label"].to(device=device)
            orig_h, orig_w = sample["orig_size"]

            # Accurate timing for GPU
            if device.type == "cuda":
                torch.cuda.synchronize(device)

            start_time = time.perf_counter()
            logits = get_semantic_logits(model_obj, image, model_category_const)

            if device.type == "cuda":
                torch.cuda.synchronize(device)

            end_time = time.perf_counter()
            total_inference_time += (end_time - start_time)

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
                current_fps = processed / total_inference_time if total_inference_time > 0 else 0.0
                print(f"Processed {processed} images | FPS: {current_fps:.2f}")

            if max_samples > 0 and processed >= max_samples:
                metrics = compute_miou_from_confmat(conf_mat)
                metrics["FPS"] = processed / total_inference_time if total_inference_time > 0 else 0.0
                metrics["Avg_Inference_Time_ms"] = (total_inference_time / processed) * 1000.0
                return metrics

    metrics = compute_miou_from_confmat(conf_mat)
    metrics["FPS"] = processed / total_inference_time if total_inference_time > 0 else 0.0
    metrics["Avg_Inference_Time_ms"] = (total_inference_time / processed) * 1000.0 if processed > 0 else 0.0
    return metrics