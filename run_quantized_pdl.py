#!/usr/bin/env python3
import argparse
import os
import random
import time
from typing import List

import torch
from torch.utils.data import DataLoader

from model.pdl import build_model
from model.quantized_conv2d import QuantizedConv2d  # keep registration side-effect

from quantization.calibration_dataset import create_calibration_loader, sample_calibration_images
from quantization.quantize_function import create_quant_sim, calibration_forward_pass

from utils.image_loader import load_images

pdl_home_path = os.path.dirname(os.path.realpath(__file__))
DEFAULT_WEIGHTS_PATH = os.path.join(pdl_home_path, "weights", "model_final_bd324a.pkl")
DEFAULT_EXPORT_PATH = os.path.join(pdl_home_path, "quantized_export")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image_height", type=int, default=512, help="input image height")
    parser.add_argument("--image_width", type=int, default=1024, help="input image width")

    parser.add_argument("--weights_path", type=str, default=DEFAULT_WEIGHTS_PATH, help="path to FP32 model weights")
    parser.add_argument(
        "--model_category",
        type=str,
        default="PANOPTIC_DEEPLAB",
        choices=["DEEPLAB_V3_PLUS", "PANOPTIC_DEEPLAB"],
        help="semantic-only or full panoptic model",
    )

    parser.add_argument(
        "--calib_images",
        type=str,
        required=True,
        help="image file or folder used for AIMET calibration",
    )

    parser.add_argument("--num_calib", type=int, default=300, help="number of calibration images")
    parser.add_argument("--batch_size", type=int, default=1, help="AIMET calibration batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="dataloader workers")
    parser.add_argument("--seed", type=int, default=123, help="random seed for calibration sampling")

    parser.add_argument("--quant_scheme", type=str, default="tf_enhanced", help="AIMET quantization scheme")
    parser.add_argument("--default_output_bw", type=int, default=8, help="activation bitwidth")
    parser.add_argument("--default_param_bw", type=int, default=8, help="parameter bitwidth")

    parser.add_argument(
        "--save_quant_checkpoint",
        type=str,
        default=None,
        help="optional path to save quantized PyTorch state_dict",
    )

    parser.add_argument("--export_path", type=str, default=DEFAULT_EXPORT_PATH, help="path to export quantized model")
    parser.add_argument("--export_prefix", type=str, default="panoptic_deeplab_int8", help="export filename prefix")
    parser.add_argument("--no_export", action="store_true", help="skip AIMET export step")

    return parser.parse_args(argv)

def main(args):
    if args.batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    if args.save_quant_checkpoint is not None:
        save_dir = os.path.dirname(args.save_quant_checkpoint)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    os.makedirs(args.export_path, exist_ok=True)

    print("Loading FP32 model...")
    model, model_category_const = build_model(
        weights_path=args.weights_path,
        model_category=args.model_category,
        image_height=args.image_height,
        image_width=args.image_width,
        device=args.device,
    )

    print("Collecting calibration images...")
    all_calib_images = load_images(args.calib_images, num_iters=-1, recursive=True)
    calib_images = sample_calibration_images(all_calib_images, args.num_calib, args.seed)
    print(f"Found {len(all_calib_images)} candidate calibration images")
    print(f"Using {len(calib_images)} images for calibration")

    calib_loader = create_calibration_loader(
        calib_image_paths=calib_images,
        image_width=args.image_width,
        image_height=args.image_height,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print("Creating AIMET QuantizationSimModel...")
    sim, _ = create_quant_sim(
        model=model,
        model_category_const=model_category_const,
        device=args.device,
        image_height=args.image_height,
        image_width=args.image_width,
        quant_scheme=args.quant_scheme,
        default_output_bw=args.default_output_bw,
        default_param_bw=args.default_param_bw,
    )

    print("Computing encodings with calibration data...")
    calib_start = time.time()
    sim.compute_encodings(
        forward_pass_callback=calibration_forward_pass,
        forward_pass_callback_args=(calib_loader, args.device),
    )
    calib_time = time.time() - calib_start
    print(f"Calibration finished in {calib_time:.2f} s")

    quantized_model = sim.model
    quantized_model.eval()

    if args.save_quant_checkpoint is not None:
        torch.save({"state_dict": quantized_model.state_dict()}, args.save_quant_checkpoint)
        print(f"Saved quantized checkpoint to: {args.save_quant_checkpoint}")

    if not args.no_export:
        print("Exporting quantized model and encodings...")
        sim.model.cpu()
        cpu_dummy_input = torch.randn(1, 3, args.image_height, args.image_width, device="cpu")

        sim.export(
            path=args.export_path,
            filename_prefix=args.export_prefix,
            dummy_input=cpu_dummy_input,
        )
        print(f"Exported files to: {args.export_path}")

    print("Done.")


if __name__ == "__main__":
    args = parse_args()
    main(args)