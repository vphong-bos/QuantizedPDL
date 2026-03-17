import torch

def extract_input(batch):
    """
    Tries common batch formats:
      - dict with 'image' or 'images'
      - tuple/list where first item may itself be dict/tensor
      - raw tensor
    """
    if isinstance(batch, dict):
        if "image" in batch:
            return extract_input(batch["image"])
        if "images" in batch:
            return extract_input(batch["images"])
        if "input" in batch:
            return extract_input(batch["input"])
        raise KeyError(f"Batch dict does not contain supported keys. Got: {list(batch.keys())}")

    if isinstance(batch, (list, tuple)):
        if len(batch) == 0:
            raise ValueError("Empty batch list/tuple.")
        return extract_input(batch[0])

    if torch.is_tensor(batch):
        return batch

    raise TypeError(f"Unsupported batch type: {type(batch)}")


def extract_tensor(output):
    """
    Tries to extract the main prediction tensor from common output formats.
    """
    if torch.is_tensor(output):
        return output

    if isinstance(output, dict):
        for key in ["semantic", "sem_logits", "logits", "out", "pred"]:
            if key in output and torch.is_tensor(output[key]):
                return output[key]
        for v in output.values():
            if torch.is_tensor(v):
                return v

    if isinstance(output, (list, tuple)):
        for v in output:
            if torch.is_tensor(v):
                return v

    raise TypeError(f"Could not extract tensor from model output type: {type(output)}")


def pearson_corrcoef(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x = x.float().reshape(-1)
    y = y.float().reshape(-1)

    x = x - x.mean()
    y = y - y.mean()

    denom = torch.sqrt((x * x).sum()) * torch.sqrt((y * y).sum())
    if denom.abs() < eps:
        return torch.tensor(0.0, device=x.device)

    return (x * y).sum() / (denom + eps)


@torch.no_grad()
def evaluate_pcc(fp32_model, quant_model, loader, device, max_samples=-1):
    fp32_model.eval()
    quant_model.eval()

    pcc_values = []
    seen = 0

    for batch in loader:
        inputs = extract_input(batch).to(device=device, dtype=torch.float32, non_blocking=True)

        fp32_out = extract_tensor(fp32_model(inputs))
        quant_out = extract_tensor(quant_model(inputs))

        if fp32_out.shape != quant_out.shape:
            raise ValueError(
                f"FP32 and quantized outputs have different shapes: "
                f"{tuple(fp32_out.shape)} vs {tuple(quant_out.shape)}"
            )

        batch_size = fp32_out.shape[0]
        for i in range(batch_size):
            pcc = pearson_corrcoef(fp32_out[i], quant_out[i])
            pcc_values.append(pcc.item())
            seen += 1

            if max_samples > 0 and seen >= max_samples:
                return {"PCC": sum(pcc_values) / len(pcc_values)}

    return {"PCC": sum(pcc_values) / len(pcc_values) if pcc_values else 0.0}