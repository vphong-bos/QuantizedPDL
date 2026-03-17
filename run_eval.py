#!/usr/bin/env python3
import argparse

import torch

from model.pdl import (
    build_model,
    load_quantized_model
)

from evaluation.eval_dataset import build_eval_loader
from evaluation.eval_metrics import evaluate_model

from utils.pcc_metric import evaluate_pcc

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cityscapes_root", type=str, required=True)

    parser.add_argument("--fp32_weights", type=str, required=True,
                        help="Path to FP32 .pkl weights")
    parser.add_argument("--quant_weights", type=str, required=True,
                        help="Path to quantized .pt/.pth weights")

    parser.add_argument("--model_category", type=str, default="PANOPTIC_DEEPLAB",
                        choices=["DEEPLAB_V3_PLUS", "PANOPTIC_DEEPLAB"])
    parser.add_argument("--image_height", type=int, default=512)
    parser.add_argument("--image_width", type=int, default=1024)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Use only first N val images, -1 for full val")

    return parser.parse_args()

def main():
    args = parse_args()

    loader = build_eval_loader(
        cityscapes_root=args.cityscapes_root,
        image_width=args.image_width,
        image_height=args.image_height,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("Loading FP32 model...")
    fp32_model, fp32_category = build_model(
        weights_path=args.fp32_weights,
        model_category=args.model_category,
        image_height=args.image_height,
        image_width=args.image_width,
        device=args.device,
    )

    print("Evaluating FP32...")
    fp32_results = evaluate_model(
        model=fp32_model,
        model_category_const=fp32_category,
        loader=loader,
        device=args.device,
        max_samples=args.max_samples,
    )

    print("Loading quantized model...")
    quant_model, quant_category = load_quantized_model(
        quant_weights=args.quant_weights,
        model_category=args.model_category,
        image_height=args.image_height,
        image_width=args.image_width,
        device=args.device,
    )

    print("Evaluating quantized...")
    quant_results = evaluate_model(
        model=quant_model,
        model_category_const=quant_category,
        loader=loader,
        device=args.device,
        max_samples=args.max_samples,
    )

    print("Evaluating PCC between FP32 and quantized outputs...")
    pcc_results = evaluate_pcc(
        fp32_model=fp32_model,
        quant_model=quant_model,
        loader=loader,
        device=args.device,
        max_samples=args.max_samples,
    )

    print("\n================ Compare FP32 vs Quantized ================")
    print(f"FP32  mIoU: {fp32_results['mIoU']:.4f}")
    print(f"INT8  mIoU: {quant_results['mIoU']:.4f}")
    print(f"Drop      : {quant_results['mIoU'] - fp32_results['mIoU']:.4f}")
    print(f"PCC       : {pcc_results['PCC']:.6f}")
    print("===========================================================")

if __name__ == "__main__":
    main()