#!/usr/bin/env python3
import argparse
import os
import random
import time
from typing import List, Tuple

import cv2
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from aimet_torch.quantsim import QuantizationSimModel

from utils.demo_utils import (
    create_deeplab_v3plus_visualization,
    create_panoptic_visualization,
    save_predictions,
)
from model.pdl import (
    DEEPLAB_V3_PLUS,
    PANOPTIC_DEEPLAB,
    PytorchPanopticDeepLab,
)

from model.quantized_conv2d import QuantizedConv2d

from quantization.calibration_dataset import CalibrationDataset
from quantization.quantize_function import create_quant_sim, calibration_forward_pass, quantize_model_with_aimet


pdl_home_path = os.path.dirname(os.path.realpath(__file__))
DEFAULT_IMAGES_PATH = os.path.join(pdl_home_path, "data/images")
DEFAULT_WEIGHTS_PATH = os.path.join(pdl_home_path, "weights", "model_final_bd324a.pkl")
DEFAULT_OUTPUT_PATH = os.path.join(pdl_home_path, "output")
DEFAULT_EXPORT_PATH = os.path.join(pdl_home_path, "quantized_export")

center_threshold = 0.05


def parse_args(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--image_height", type=int, default=512, help="input image height")
    parser.add_argument("--image_width", type=int, default=1024, help="input image width")

    parser.add_argument("--weights_path", type=str, default=DEFAULT_WEIGHTS_PATH, help="path to model weights")
    parser.add_argument("--model_category",
                        type=str,
                        default="PANOPTIC_DEEPLAB",
                        choices=["DEEPLAB_V3_PLUS", "PANOPTIC_DEEPLAB"],
                        help="semantic-only or full panoptic model")

    parser.add_argument("--calib_images",
                        type=str,
                        required=True,
                        help="image file or folder used for AIMET calibration")
    parser.add_argument("--eval_images",
                        type=str,
                        default=None,
                        help="optional image file or folder for post-quant inference")
    parser.add_argument("--output_path", type=str, default=DEFAULT_OUTPUT_PATH, help="path to save visual outputs")
    parser.add_argument("--export_path", type=str, default=DEFAULT_EXPORT_PATH, help="path to export quantized model")
    parser.add_argument("--export_prefix", type=str, default="panoptic_deeplab_int8", help="export filename prefix")

    parser.add_argument("--num_calib", type=int, default=300, help="number of calibration images")
    parser.add_argument("--num_eval", type=int, default=20, help="number of eval images to run after quantization")
    parser.add_argument("--batch_size", type=int, default=1, help="AIMET calibration batch size")
    parser.add_argument("--num_workers", type=int, default=2, help="dataloader workers")
    parser.add_argument("--seed", type=int, default=123, help="random seed for calibration sampling")

    parser.add_argument("--quant_scheme", type=str, default="tf_enhanced", help="AIMET quantization scheme")
    parser.add_argument("--default_output_bw", type=int, default=8, help="activation bitwidth")
    parser.add_argument("--default_param_bw", type=int, default=8, help="parameter bitwidth")

    parser.add_argument("--save_visualizations", action="store_true", help="save visualizations for eval images")
    parser.add_argument("--no_export", action="store_true", help="skip AIMET export step")

    return parser.parse_args(argv)


def load_images(images_path: str, num_iters: int = -1, recursive: bool = True) -> List[str]:
    image_extensions = {".png", ".jpg", ".jpeg"}

    if images_path is None:
        return []

    if os.path.isfile(images_path):
        images = [images_path]
    else:
        images = []
        if recursive:
            for root, _, files in os.walk(images_path):
                for file in sorted(files):
                    _, ext = os.path.splitext(file)
                    if ext.lower() in image_extensions:
                        images.append(os.path.join(root, file))
        else:
            for file in sorted(os.listdir(images_path)):
                full_path = os.path.join(images_path, file)
                _, ext = os.path.splitext(file)
                if os.path.isfile(full_path) and ext.lower() in image_extensions:
                    images.append(full_path)

    images = sorted(images)
    if num_iters != -1:
        images = images[:num_iters]
    return images


def build_transform():
    # Match your current inference preprocessing.
    # If the original training pipeline used normalization, add it here too.
    return T.Compose([
        T.ToTensor(),
    ])

def preprocess_image_from_path(image_path: str, input_width: int, input_height: int, device: str):
    transform = build_transform()

    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Failed to read image: {image_path}")

    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image = cv2.resize(original_image, (input_width, input_height))

    torch_input = transform(Image.fromarray(original_image)).unsqueeze(0).to(device=device, dtype=torch.float32)
    return original_image, torch_input


def build_model(weights_path: str, model_category: str, image_height: int, image_width: int, device: str):
    model_category_const = PANOPTIC_DEEPLAB if model_category == "PANOPTIC_DEEPLAB" else DEEPLAB_V3_PLUS

    model = PytorchPanopticDeepLab(
        num_classes=19,
        common_stride=4,
        project_channels=[32, 64],
        decoder_channels=[256, 256, 256],
        sem_seg_head_channels=256,
        ins_embed_head_channels=32,
        train_size=(image_height, image_width),
        weights_path=weights_path,
        model_category=model_category_const,
    )

    model = model.to(device=device, dtype=torch.float32)
    model.eval()
    return model, model_category_const


def run_inference(model, torch_input, model_category_const):
    with torch.no_grad():
        outputs = model(torch_input)

    if model_category_const == DEEPLAB_V3_PLUS:
        if isinstance(outputs, (tuple, list)):
            return outputs[0]
        return outputs

    # PANOPTIC_DEEPLAB
    if not isinstance(outputs, (tuple, list)):
        raise TypeError(f"Expected tuple/list output, got {type(outputs)}")

    if len(outputs) >= 3:
        semantic_logits = outputs[0]
        center_heatmap = outputs[1]
        offset_map = outputs[2]
        return semantic_logits, center_heatmap, offset_map

    raise ValueError(f"Unexpected number of outputs: {len(outputs)}")


def save_visualization(model_category_const, output, original_image, output_path, image_path):
    image_name = os.path.basename(image_path)
    image_stem = os.path.splitext(image_name)[0]
    output_dir = os.path.join(output_path, f"{image_stem}_output")
    os.makedirs(output_dir, exist_ok=True)

    if model_category_const == DEEPLAB_V3_PLUS:
        semantic_logits = output
        semantic_np = semantic_logits.float().squeeze(0).permute(1, 2, 0).cpu().numpy()

        vis, _ = create_deeplab_v3plus_visualization(
            semantic_np,
            original_image=original_image,
        )
    else:
        semantic_logits, center_heatmap, offset_map = output

        semantic_np = semantic_logits.float().squeeze(0).permute(1, 2, 0).cpu().numpy()
        center_np = center_heatmap.float().squeeze(0).permute(1, 2, 0).cpu().numpy()
        offset_np = offset_map.float().squeeze(0).permute(1, 2, 0).cpu().numpy()

        vis, _ = create_panoptic_visualization(
            semantic_np,
            center_np,
            offset_np,
            original_image,
            center_threshold=center_threshold,
            score_threshold=center_threshold,
            stuff_area=1,
            top_k=1000,
            nms_kernel=11,
        )

    save_predictions(output_dir, image_name, original_image, vis)
    print(f"Saved output for image {image_name} to {output_dir}")

def create_calibration_loader(calib_image_paths: List[str],
                              image_width: int,
                              image_height: int,
                              batch_size: int,
                              num_workers: int):
    dataset = CalibrationDataset(
        image_paths=calib_image_paths,
        image_width=image_width,
        image_height=image_height,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    return loader

def sample_calibration_images(all_image_paths: List[str], num_calib: int, seed: int) -> List[str]:
    if len(all_image_paths) == 0:
        raise ValueError("No calibration images found.")

    if num_calib >= len(all_image_paths):
        return all_image_paths

    rng = random.Random(seed)
    sampled = rng.sample(all_image_paths, num_calib)
    sampled.sort()
    return sampled

def run_post_quant_eval(model, model_category_const, eval_image_paths: List[str],
                        image_width: int, image_height: int, device: str,
                        output_path: str, save_visualizations_flag: bool):
    if len(eval_image_paths) == 0:
        print("No eval images provided. Skipping post-quant inference.")
        return

    os.makedirs(output_path, exist_ok=True)

    total_start = time.time()
    for i, image_path in enumerate(eval_image_paths, start=1):
        original_image, torch_input = preprocess_image_from_path(
            image_path=image_path,
            input_width=image_width,
            input_height=image_height,
            device=device,
        )

        start_time = time.time()
        output = run_inference(model, torch_input, model_category_const)
        end_time = time.time()

        print(f"[eval {i}/{len(eval_image_paths)}] {os.path.basename(image_path)}: {(end_time - start_time) * 1000:.2f} ms")

        if save_visualizations_flag:
            save_visualization(
                model_category_const=model_category_const,
                output=output,
                original_image=original_image,
                output_path=output_path,
                image_path=image_path,
            )

    total_time = time.time() - total_start
    fps = len(eval_image_paths) / total_time if total_time > 0 else 0.0
    print("\n================ Post-Quant Inference Results ================")
    print(f"Model Category: {'PANOPTIC_DEEPLAB' if model_category_const == PANOPTIC_DEEPLAB else 'DEEPLAB_V3_PLUS'}")
    print(f"Number of Inputs: {len(eval_image_paths)}")
    print(f"Total Execution Time: {total_time:.4f} s")
    print(f"Samples per Second: {fps:.2f} samples/s")
    print("==============================================================")


def main(args):
    if args.batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    os.makedirs(args.output_path, exist_ok=True)
    os.makedirs(args.export_path, exist_ok=True)

    print("Loading model...")
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
    sim, dummy_input = create_quant_sim(
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

    if args.eval_images is not None:
        eval_image_paths = load_images(args.eval_images, num_iters=args.num_eval, recursive=True)
        run_post_quant_eval(
            model=quantized_model,
            model_category_const=model_category_const,
            eval_image_paths=eval_image_paths,
            image_width=args.image_width,
            image_height=args.image_height,
            device=args.device,
            output_path=args.output_path,
            save_visualizations_flag=args.save_visualizations,
        )

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