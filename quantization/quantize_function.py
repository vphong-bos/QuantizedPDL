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
import json
import math
import torch
from aimet_torch.quantsim import QuantizationSimModel

def _num_expected_encodings_from_quantizer(quantizer):
    """
    Return how many legacy encodings this quantizer expects.
    For scalar/per-tensor quantizers, expect 1.
    For per-channel quantizers, expect product(shape).
    """
    shape = getattr(quantizer, "shape", None)

    if shape is None:
        return 1

    try:
        # torch.Size([]) -> scalar quantizer -> expect 1
        if len(shape) == 0:
            return 1

        n = 1
        for x in shape:
            n *= int(x)
        return max(n, 1)
    except Exception:
        return 1


def _build_param_encoding_expectation_map(sim_model):
    """
    Build mapping:
        full_param_name -> expected_number_of_legacy_encoding_entries
    Example:
        'model.backbone.stem.conv1.norm.weight' -> 1
        'model.some_conv.weight' -> 64
    """
    expected = {}

    for module_name, module in sim_model.named_modules():
        param_quantizers = getattr(module, "param_quantizers", None)
        if not param_quantizers:
            continue

        # AIMET commonly exposes this as a dict-like mapping
        for param_name, quantizer in param_quantizers.items():
            if quantizer is None:
                continue

            full_name = f"{module_name}.{param_name}" if module_name else param_name
            expected[full_name] = _num_expected_encodings_from_quantizer(quantizer)

    return expected


def load_aimet_quantized_model(
    quant_weights,
    encoding_path,
    model_category,
    image_height,
    image_width,
    device,
    quant_scheme="tf_enhanced",
    default_output_bw=8,
    default_param_bw=8,
):
    print("Loading quantized model...")

    if not isinstance(encoding_path, (str, bytes, os.PathLike)):
        raise ValueError(f"encoding_path must be a valid path, got: {encoding_path} ({type(encoding_path)})")
    if not os.path.exists(encoding_path):
        raise FileNotFoundError(f"Encoding file not found: {encoding_path}")
    if not os.path.exists(quant_weights):
        raise FileNotFoundError(f"Quantized weights file not found: {quant_weights}")

    loaded_obj = torch.load(quant_weights, map_location=device, weights_only=False)

    # Extract state_dict once
    # ------------------------------------------------------------
    # Normalize loaded artifact into a state_dict
    # ------------------------------------------------------------
    if isinstance(loaded_obj, dict):
        if "state_dict" in loaded_obj:
            state_dict = loaded_obj["state_dict"]
        elif "model_state_dict" in loaded_obj:
            state_dict = loaded_obj["model_state_dict"]
        elif "model" in loaded_obj and hasattr(loaded_obj["model"], "state_dict"):
            print("[load] Checkpoint contains model object in key 'model'; extracting state_dict")
            state_dict = loaded_obj["model"].state_dict()
        else:
            state_dict = loaded_obj
    elif hasattr(loaded_obj, "state_dict"):
        print(f"[load] Loaded object of type {type(loaded_obj)}; extracting state_dict")
        state_dict = loaded_obj.state_dict()
    else:
        raise ValueError(f"Unsupported quant_weights object type: {type(loaded_obj)}")

    if not isinstance(state_dict, dict):
        raise ValueError(f"Loaded state_dict is not a dict: {type(state_dict)}")

    # Recreate model + sim
    if not isinstance(state_dict, dict):
        raise ValueError(f"Loaded state_dict is not a dict: {type(state_dict)}")

    # ------------------------------------------------------------
    # Recreate float model + AIMET sim
    # ------------------------------------------------------------
    model, model_category_const = build_model(
        weights_path=None,
        model_category=model_category,
        image_height=image_height,
        image_width=image_width,
        device=device,
    )
    model.eval()

    wrapped_model = AimetTraceWrapper(model, model_category_const).to(device).eval()

    dummy_input = torch.randn(1, 3, image_height, image_width, device=device)
    sim = QuantizationSimModel(
        model=wrapped_model,
        dummy_input=dummy_input,
        quant_scheme=quant_scheme,
        default_output_bw=default_output_bw,
        default_param_bw=default_param_bw,
    )

    # Fix keys to match sim.model
    # ------------------------------------------------------------
    # Normalize checkpoint keys to match sim.model
    # ------------------------------------------------------------
    target_state = sim.model.state_dict()
    target_keys = set(target_state.keys())

    fixed_sd = {}
    dropped_ckpt_keys = []

    for k, v in state_dict.items():
        nk = k

        if nk.startswith("module."):
            nk = nk[len("module."):]

        if nk.startswith("wrapped_model."):
            nk = nk[len("wrapped_model."):]

        if not nk.startswith("model."):
            nk = "model." + nk

        if nk in target_keys:
            fixed_sd[nk] = v
        else:
            dropped_ckpt_keys.append(nk)

    if dropped_ckpt_keys:
        print(f"[load] dropped {len(dropped_ckpt_keys)} unmatched checkpoint keys")
        print("[load] first dropped checkpoint keys:", dropped_ckpt_keys[:10])

    missing, unexpected = sim.model.load_state_dict(fixed_sd, strict=False)
    print(f"[load] missing keys: {len(missing)}")
    print(f"[load] unexpected keys: {len(unexpected)}")
    if missing:
        print("[load] first missing keys:", missing[:10])
    if unexpected:
        print("[load] first unexpected keys:", unexpected[:10])

    sim.set_and_freeze_param_encodings(encoding_path)
    sim.set_and_freeze_activation_encodings(encoding_path)
    print(f"[load] Loaded encodings from: {encoding_path}")
    # ------------------------------------------------------------
    # Load and filter encoding JSON
    # ------------------------------------------------------------
    with open(encoding_path, "r") as f:
        enc = json.load(f)

    if not isinstance(enc, dict):
        raise ValueError(f"Encoding JSON root must be dict, got: {type(enc)}")

    print("[load] encoding top-level keys:", list(enc.keys()))

    valid_param_names = set(sim.model.state_dict().keys())
    valid_module_names = {name for name, _ in sim.model.named_modules()}
    expected_param_encoding_counts = _build_param_encoding_expectation_map(sim.model)

    removed_param_encodings = []
    removed_activation_encodings = []

    # Filter parameter encodings by:
    #   1) name exists
    #   2) number of encoding entries matches quantizer expectation
    if "param_encodings" in enc:
        if not isinstance(enc["param_encodings"], dict):
            raise ValueError(
                f"Expected 'param_encodings' to be dict, got: {type(enc['param_encodings'])}"
            )

        filtered_param_encodings = {}

        for k, v in enc["param_encodings"].items():
            if k not in valid_param_names:
                removed_param_encodings.append((k, "missing_param"))
                continue

            expected_count = expected_param_encoding_counts.get(k, 1)

            # Legacy encodings are usually a list of dicts
            if isinstance(v, list):
                actual_count = len(v)
            else:
                actual_count = 1

            if actual_count != expected_count:
                removed_param_encodings.append(
                    (k, f"count_mismatch expected={expected_count} actual={actual_count}")
                )
                continue

            filtered_param_encodings[k] = v

        print(
            f"[load] param encodings kept: {len(filtered_param_encodings)} / {len(enc['param_encodings'])}"
        )
        if removed_param_encodings:
            print("[load] first removed param encodings:", removed_param_encodings[:10])

        enc["param_encodings"] = filtered_param_encodings

    # Filter activation encodings only by name for now
    if "activation_encodings" in enc:
        if not isinstance(enc["activation_encodings"], dict):
            raise ValueError(
                f"Expected 'activation_encodings' to be dict, got: {type(enc['activation_encodings'])}"
            )

        filtered_activation_encodings = {}
        for k, v in enc["activation_encodings"].items():
            if k in valid_module_names or k in valid_param_names:
                filtered_activation_encodings[k] = v
            else:
                removed_activation_encodings.append(k)

        print(
            f"[load] activation encodings kept: {len(filtered_activation_encodings)} / {len(enc['activation_encodings'])}"
        )
        if removed_activation_encodings:
            print("[load] first removed activation encodings:", removed_activation_encodings[:10])

        enc["activation_encodings"] = filtered_activation_encodings

    filtered_encoding_path = os.path.splitext(str(encoding_path))[0] + ".filtered.encodings"
    with open(filtered_encoding_path, "w") as f:
        json.dump(enc, f, indent=2)

    print(f"[load] wrote filtered encodings to: {filtered_encoding_path}")

    # ------------------------------------------------------------
    # Load filtered encodings
    # ------------------------------------------------------------
    # Load encodings according to AIMET version
    if hasattr(sim, "set_and_freeze_activation_encodings"):
        sim.set_and_freeze_param_encodings(filtered_encoding_path)
        sim.set_and_freeze_activation_encodings(filtered_encoding_path)
    else:
        # Older / different AIMET versions only expose load_encodings()
        sim.load_encodings(filtered_encoding_path)

    print(f"[load] Loaded encodings from: {filtered_encoding_path}")
    sim.model.eval()
    return sim.model, model_category_const
