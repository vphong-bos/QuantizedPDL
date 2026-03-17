import torch
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
    # 1. Load checkpoint
    ckpt = torch.load(quant_weights, map_location=device, weights_only=False)

    if not isinstance(ckpt, dict):
        raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")

    state_dict = ckpt.get("state_dict", ckpt)
    encoding_path = ckpt.get("encoding_path", None)
    best_score = ckpt.get("best_score", None)

    print(f"[load] encoding_path from checkpoint: {encoding_path}")
    print(f"[load] best_score from checkpoint: {best_score}")

    # 2. Rebuild FP32 model
    model, model_category_const = build_model(
        weights_path=None,
        model_category=model_category,
        image_height=image_height,
        image_width=image_width,
        device=device,
    )
    model.eval()

    # 3. Wrap model for AIMET-safe forward
    wrapped_model = AimetTraceWrapper(model, model_category_const).to(device).eval()

    # 4. Recreate QuantSim
    dummy_input = torch.randn(1, 3, image_height, image_width, device=device)
    sim = QuantizationSimModel(
        model=wrapped_model,
        dummy_input=dummy_input,
        quant_scheme=quant_scheme,
        default_output_bw=default_output_bw,
        default_param_bw=default_param_bw,
    )

    # 5. Clean prefixes and load weights
    cleaned_sd = {}
    for k, v in state_dict.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        cleaned_sd[nk] = v

    missing, unexpected = sim.model.load_state_dict(cleaned_sd, strict=False)
    print(f"[load] missing keys: {len(missing)}")
    print(f"[load] unexpected keys: {len(unexpected)}")
    if missing:
        print("[load] first missing keys:", missing[:10])
    if unexpected:
        print("[load] first unexpected keys:", unexpected[:10])

    # 6. Load encodings if checkpoint contains them
    if encoding_path is not None:
        sim.set_and_freeze_param_encodings(encoding_path)
        sim.set_and_freeze_activation_encodings(encoding_path)
        print(f"[load] Loaded encodings from: {encoding_path}")
    else:
        print("[load] No encoding_path found in checkpoint; loaded weights only.")

    sim.model.eval()
    return sim.model, model_category_const