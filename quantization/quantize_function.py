import torch
from torch.utils.data import DataLoader, Dataset
from aimet_torch.quantsim import QuantizationSimModel
import torchvision.transforms as T
from quantization.calibration_dataset import CalibrationDataset

def create_quant_sim(model, device, image_height, image_width):
    dummy_input = torch.randn(1, 3, image_height, image_width, device=device)

    sim = QuantizationSimModel(
        model=model,
        dummy_input=dummy_input,
        quant_scheme="tf_enhanced",
        default_output_bw=8,
        default_param_bw=8,
    )
    return sim

def calibration_forward_pass(model, forward_args):
    dataloader, device, max_batches = forward_args
    model.eval()
    with torch.no_grad():
        for batch_idx, images in enumerate(dataloader):
            images = images.to(device)
            _ = model(images)

            if max_batches is not None and (batch_idx + 1) >= max_batches:
                break

def quantize_model_with_aimet(model, image_paths, device, image_height, image_width, num_calib=300):
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

    sim = create_quant_sim(model, device, image_height, image_width)

    sim.compute_encodings(
        forward_pass_callback=calibration_forward_pass,
        forward_pass_callback_args=(loader, device, None),
    )

    return sim

