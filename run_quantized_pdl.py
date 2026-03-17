#!/usr/bin/env python3
import argparse
import os
import time

import torch

from model.pdl import build_model

from quantization.calibration_dataset import create_calibration_loader, sample_calibration_images
from quantization.quantize_function import AimetTraceWrapper

from aimet_torch.cross_layer_equalization import equalize_model
from aimet_torch.auto_quant import AutoQuant

from utils.image_loader import load_images

from evaluation.eval_dataset import build_eval_loader
from evaluation.eval_metrics import evaluate_model

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

    parser.add_argument("--num_calib", type=int, default=800, help="number of calibration images")
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

    # AutoQuant-specific
    parser.add_argument("--allowed_accuracy_drop", type=float, default=0.01,
                        help="allowed accuracy drop for AutoQuant")
    parser.add_argument("--results_dir", type=str, default=None,
                        help="AutoQuant results dir; defaults to export_path/autoquant_results")
    parser.add_argument("--cityscapes_root", type=str, required=True,
                        help="Cityscapes root containing leftImg8bit/val and gtFine/val")
    return parser.parse_args(argv)


def main(args):
    if args.batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    if args.save_quant_checkpoint is not None:
        save_dir = os.path.dirname(args.save_quant_checkpoint)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    os.makedirs(args.export_path, exist_ok=True)

    results_dir = args.results_dir
    if results_dir is None:
        results_dir = os.path.join(args.export_path, "autoquant_results")
    os.makedirs(results_dir, exist_ok=True)

    print("Loading FP32 model...")
    model, model_category_const = build_model(
        weights_path=args.weights_path,
        model_category=args.model_category,
        image_height=args.image_height,
        image_width=args.image_width,
        device=args.device,
    )

    print("Applying Cross-Layer Equalization...")
    wrapped_model = AimetTraceWrapper(model, model_category_const).eval()

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
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    print("Building validation loader...")
    eval_loader = build_eval_loader(
        cityscapes_root=args.cityscapes_root,
        image_width=args.image_width,
        image_height=args.image_height,
        batch_size=1,
        num_workers=args.num_workers,
    )

    print("Creating AutoQuant...")
    dummy_input = torch.randn(1, 3, args.image_height, args.image_width, device=args.device)

    def eval_callback(candidate_model, num_samples=None):
        candidate_model.eval()
        results = evaluate_model(
            model=candidate_model,
            model_category_const=model_category_const,
            loader=eval_loader,
            device=args.device,
            max_samples=-1 if num_samples is None else num_samples,
        )
        return results["mIoU"]

    auto_quant = AutoQuant(
        model=wrapped_model,
        dummy_input=dummy_input,
        data_loader=calib_loader,
        eval_callback=eval_callback,
        param_bw=args.default_param_bw,
        output_bw=args.default_output_bw,
        quant_scheme=args.quant_scheme,
        results_dir=results_dir,
    )

    print("Running AutoQuant...")
    aq_start = time.time()
    best_model, _, encoding_path, _ = auto_quant.optimize(
        allowed_accuracy_drop=args.allowed_accuracy_drop
    )
    aq_time = time.time() - aq_start
    print(f"AutoQuant finished in {aq_time:.2f} s")

    quantized_model = best_model
    quantized_model.eval()

    if args.save_quant_checkpoint is not None:
        torch.save({"state_dict": quantized_model.state_dict()}, args.save_quant_checkpoint)
        print(f"Saved quantized checkpoint to: {args.save_quant_checkpoint}")

    if not args.no_export:
        print("Exporting quantized model and encodings...")
        quantized_model.cpu().eval()
        cpu_dummy_input = torch.randn(1, 3, args.image_height, args.image_width, device="cpu")

        if args.model_category == "PANOPTIC_DEEPLAB":
            output_names = ["semantic_logits", "center_heatmap", "offset_map"]
        else:
            output_names = ["semantic_logits"]

        onnx_path = os.path.join(args.export_path, f"{args.export_prefix}.onnx")
        torch.onnx.export(
            quantized_model,
            cpu_dummy_input,
            onnx_path,
            export_params=True,
            opset_version=13,
            do_constant_folding=True,
            input_names=["input"],
            output_names=output_names,
        )
        print(f"Exported ONNX to: {onnx_path}")

    print("Done.")

if __name__ == "__main__":
    args = parse_args()
    main(args)