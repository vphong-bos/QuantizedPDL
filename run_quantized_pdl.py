#!/usr/bin/env python3
import argparse
import os
import time

import torch

from model.pdl import build_model

from quantization.calibration_dataset import create_calibration_loader, sample_calibration_images
from quantization.quantize_function import AimetTraceWrapper, create_quant_sim, calibration_forward_pass
from utils.image_loader import load_images

from aimet_torch.batch_norm_fold import fold_all_batch_norms
from aimet_torch.adaround.adaround_weight import Adaround, AdaroundParameters
from aimet_torch import quantsim

pdl_home_path = os.path.dirname(os.path.realpath(__file__))
DEFAULT_WEIGHTS_PATH = os.path.join(pdl_home_path, "weights", "model_final_bd324a.pkl")
DEFAULT_EXPORT_PATH = os.path.join(pdl_home_path, "quantized_export")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image_height", type=int, default=1024, help="input image height")
    parser.add_argument("--image_width", type=int, default=2048, help="input image width")

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
        help="optional path to save AIMET sim checkpoint",
    )

    parser.add_argument("--export_path", type=str, default=DEFAULT_EXPORT_PATH, help="path to export quantized model")
    parser.add_argument("--export_prefix", type=str, default="panoptic_deeplab_int8", help="export filename prefix")
    parser.add_argument("--no_export", action="store_true", help="skip AIMET export step")

    parser.add_argument(
        "--config_file",
        type=str,
        default=None,
        help="Config for quantize model",
    )

    parser.add_argument(
        "--enable_cle",
        dest="enable_cle",
        action="store_true",
        help="enable Cross-Layer Equalization before quantization",
    )
    parser.add_argument(
        "--disable_cle",
        dest="enable_cle",
        action="store_false",
        help="disable Cross-Layer Equalization",
    )
    parser.set_defaults(enable_cle=False)

    parser.add_argument(
        "--enable_bn_fold",
        action="store_true",
        help="apply batch norm folding before creating QuantSim",
    )

    parser.add_argument(
        "--enable_adaround",
        action="store_true",
        help="apply AIMET AdaRound before creating QuantSim",
    )
    parser.add_argument(
        "--adaround_num_batches",
        type=int,
        default=32,
        help="number of calibration batches to use for AdaRound",
    )
    parser.add_argument(
        "--adaround_num_iterations",
        type=int,
        default=10000,
        help="number of iterations for AdaRound",
    )
    parser.add_argument(
        "--adaround_path",
        type=str,
        default=None,
        help="directory to save AdaRound encodings",
    )
    parser.add_argument(
        "--adaround_prefix",
        type=str,
        default="adaround",
        help="filename prefix for AdaRound encodings",
    )

    return parser.parse_args(argv)


def adaround_forward_fn(model, inputs):
    if isinstance(inputs, dict):
        images = inputs["image"]
    elif isinstance(inputs, (list, tuple)):
        images = inputs[0]
    else:
        images = inputs

    images = images.to(next(model.parameters()).device)
    return model(images)


def main(args):
    if args.batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    if args.save_quant_checkpoint is not None:
        save_dir = os.path.dirname(args.save_quant_checkpoint)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

    os.makedirs(args.export_path, exist_ok=True)

    if args.adaround_path is None:
        args.adaround_path = args.export_path
    os.makedirs(args.adaround_path, exist_ok=True)

    print("Loading FP32 model...")
    model, model_category_const = build_model(
        weights_path=args.weights_path,
        model_category=args.model_category,
        image_height=args.image_height,
        image_width=args.image_width,
        device=args.device,
    )
    model = model.to(args.device).eval()

    if args.enable_cle:
        print("Applying Cross-Layer Equalization (CLE)...")
        from aimet_torch.cross_layer_equalization import equalize_model

        cle_start = time.time()

        model = model.cpu().eval()
        dummy_input_cpu = torch.randn(1, 3, args.image_height, args.image_width, device="cpu")

        cle_wrapper = AimetTraceWrapper(
            model=model,
            model_category_const=model_category_const,
        ).cpu().eval()

        equalize_model(
            cle_wrapper,
            input_shapes=(1, 3, args.image_height, args.image_width),
            dummy_input=dummy_input_cpu,
        )

        model = model.to(args.device).eval()
        cle_time = time.time() - cle_start
        print(f"CLE finished in {cle_time:.2f} s")
    else:
        print("CLE disabled")

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

    print("Wrapping model for AIMET tracing...")
    wrapped_model = AimetTraceWrapper(
        model=model,
        model_category_const=model_category_const,
    ).cpu().eval()

    if args.enable_bn_fold:
        print("Applying batch norm folding...")
        dummy_input_cpu = torch.randn(1, 3, args.image_height, args.image_width, device="cpu")

        fold_all_batch_norms(
            model=wrapped_model,
            input_shapes=(1, 3, args.image_height, args.image_width),
            dummy_input=dummy_input_cpu,
        )

    wrapped_model = wrapped_model.to(args.device).eval()

    if args.enable_adaround:
        print("Applying AdaRound...")
        dummy_input_cpu = torch.randn(1, 3, args.image_height, args.image_width, device="cpu")

        adaround_params = AdaroundParameters(
            data_loader=calib_loader,
            num_batches=min(args.adaround_num_batches, len(calib_loader)),
            default_num_iterations=args.adaround_num_iterations,
            forward_fn=adaround_forward_fn,
        )

        wrapped_model = wrapped_model.cpu().eval()
        wrapped_model = Adaround.apply_adaround(
            model=wrapped_model,
            dummy_input=dummy_input_cpu,
            params=adaround_params,
            path=args.adaround_path,
            filename_prefix=args.adaround_prefix,
            default_param_bw=args.default_param_bw,
            default_quant_scheme=args.quant_scheme,
            default_config_file=args.config_file,
        )

        wrapped_model = wrapped_model.to(args.device).eval()
        print(
            f"AdaRound finished. Encodings saved under: "
            f"{os.path.join(args.adaround_path, args.adaround_prefix)}*.encodings"
        )
    else:
        print("AdaRound disabled")

    print("Creating AIMET QuantizationSimModel...")
    sim, _ = create_quant_sim(
        model=wrapped_model,
        model_category_const=model_category_const,
        device=args.device,
        image_height=args.image_height,
        image_width=args.image_width,
        quant_scheme=args.quant_scheme,
        default_output_bw=args.default_output_bw,
        default_param_bw=args.default_param_bw,
        config_file=args.config_file,
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
        quantsim.save_checkpoint(sim, args.save_quant_checkpoint)
        print(f"Saved AIMET sim checkpoint to: {args.save_quant_checkpoint}")

    if not args.no_export:
        print("Exporting quantized model and encodings...")
        sim.model.cpu().eval()
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