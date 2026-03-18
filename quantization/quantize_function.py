import torch
import os
from torch.utils.data import DataLoader, Dataset
from aimet_torch.quantsim import QuantizationSimModel
import torchvision.transforms as T
from quantization.calibration_dataset import CalibrationDataset

from model.quantized_conv2d import QuantizedConv2d
from model.pdl import build_model

import torch
import torch.nn as nn

from model.pdl import DEEPLAB_V3_PLUS, PANOPTIC_DEEPLAB

class AimetTraceWrapper(nn.Module):
    def __init__(self, model, model_category_const):
        super().__init__()
        self.model = model
        self.model_category_const = model_category_const

    def forward(self, x):
        outputs = self.model(x)

        # Expecting original model to return:
        # semantic_logits, center_heatmap, offset_map, something_else
        if self.model_category_const == DEEPLAB_V3_PLUS:
            if isinstance(outputs, (tuple, list)):
                semantic_logits = outputs[0]
            else:
                semantic_logits = outputs
            return semantic_logits

        # PANOPTIC_DEEPLAB
        if not isinstance(outputs, (tuple, list)):
            raise TypeError(f"Expected tuple/list output from model, got {type(outputs)}")

        semantic_logits = outputs[0]
        center_heatmap = outputs[1]
        offset_map = outputs[2]

        # Return only tensors
        return semantic_logits, center_heatmap, offset_map

def create_quant_sim(
    model,
    model_category_const,
    device,
    image_height,
    image_width,
    quant_scheme,
    default_output_bw,
    default_param_bw,
    config_file,
):
    dummy_input = torch.randn(1, 3, image_height, image_width, device=device)

    wrapped_model = AimetTraceWrapper(model, model_category_const).to(device)
    wrapped_model.eval()

    sim = QuantizationSimModel(
        model=wrapped_model,
        dummy_input=dummy_input,
        quant_scheme=quant_scheme,
        default_output_bw=default_output_bw,
        default_param_bw=default_param_bw,
        config_file = config_file,
    )

    return sim, dummy_input


def calibration_forward_pass(model, forward_pass_args):
    dataloader, device = forward_pass_args
    model.eval()
    with torch.no_grad():
        for images in dataloader:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            _ = model(images)

def quantize_model_with_aimet(
    model,
    model_category_const,
    image_paths,
    device,
    image_height,
    image_width,
    num_calib=500,
    quant_scheme="tf_enhanced",
    default_output_bw=8,
    default_param_bw=8,
):
    dataset = CalibrationDataset(
        image_paths=image_paths[:num_calib],
        image_width=image_width,
        image_height=image_height,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    sim, _ = create_quant_sim(
        model=model,
        model_category_const=model_category_const,
        device=device,
        image_height=image_height,
        image_width=image_width,
        quant_scheme=quant_scheme,
        default_output_bw=default_output_bw,
        default_param_bw=default_param_bw,
    )

    sim.compute_encodings(
        forward_pass_callback=calibration_forward_pass,
        forward_pass_callback_args=(loader, device),
    )

    return sim

import os

def load_aimet_quantized_model(
    quant_weights,
    model_category,
    image_height,
    image_width,
    device,
    quant_scheme="tf_enhanced",
    default_output_bw=8,
    default_param_bw=8,
):
    print("Loading quantized model from .pth only...")

    if not os.path.exists(quant_weights):
        raise FileNotFoundError(f"Quantized weights file not found: {quant_weights}")

    loaded_obj = torch.load(quant_weights, map_location=device, weights_only=False)

    model_category_const = (
        PANOPTIC_DEEPLAB if model_category == "PANOPTIC_DEEPLAB" else DEEPLAB_V3_PLUS
    )

    # Case 1: saved full model / wrapper object
    if hasattr(loaded_obj, "to") and hasattr(loaded_obj, "eval"):
        print(f"[load] Loaded model object directly: {type(loaded_obj)}")
        model = loaded_obj.to(device).eval()
        return model, model_category_const

    # Case 2: checkpoint/state_dict only -> rebuild model and load weights
    if isinstance(loaded_obj, dict):
        if "state_dict" in loaded_obj:
            state_dict = loaded_obj["state_dict"]
        elif "model_state_dict" in loaded_obj:
            state_dict = loaded_obj["model_state_dict"]
        elif "model" in loaded_obj and hasattr(loaded_obj["model"], "to"):
            print("[load] Loaded model object from checkpoint['model']")
            model = loaded_obj["model"].to(device).eval()
            return model, model_category_const
        else:
            state_dict = loaded_obj

        model, _ = build_model(
            weights_path=None,
            model_category=model_category,
            image_height=image_height,
            image_width=image_width,
            device=device,
            quant_scheme="tf_enhanced",
            default_output_bw=8,
            default_param_bw=8,
        )
        model = model.to(device).eval()

        cleaned_sd = {}
        target_keys = set(model.state_dict().keys())

        for k, v in state_dict.items():
            nk = k
            if nk.startswith("module."):
                nk = nk[len("module."):]
            if nk in target_keys:
                cleaned_sd[nk] = v

        missing, unexpected = model.load_state_dict(cleaned_sd, strict=False)
        print(f"[load] missing keys: {len(missing)}")
        print(f"[load] unexpected keys: {len(unexpected)}")
        if missing:
            print("[load] first missing keys:", missing[:10])
        if unexpected:
            print("[load] first unexpected keys:", unexpected[:10])

        return model, model_category_const

    raise ValueError(f"Unsupported quant_weights object type: {type(loaded_obj)}")
