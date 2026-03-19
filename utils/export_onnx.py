import os
import onnxruntime as ort

def export_optimized_onnx_model(
    quant_weights,
    output_path=None,
    provider="CPUExecutionProvider",
):
    """
    Export an optimized ONNX model using ONNX Runtime graph optimization.

    Args:
        quant_weights (str): Path to input ONNX model.
        output_path (str|None): Where to save optimized model.
                                If None, auto-generates '<input>.optimized.onnx'.
        provider (str): ONNX Runtime provider.

    Returns:
        str: Saved optimized model path.
    """
    ext = os.path.splitext(quant_weights)[1].lower()
    if ext != ".onnx":
        raise ValueError(f"export_optimized_onnx_model expects an .onnx file, got: {quant_weights}")

    if output_path is None:
        base, _ = os.path.splitext(quant_weights)
        output_path = base + ".optimized.onnx"

    print(f"Exporting optimized ONNX model from: {quant_weights}")
    print(f"Saving to: {output_path}")
    print(f"Using provider: {provider}")

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.optimized_model_filepath = output_path

    # Creating the session triggers optimization + saving
    _ = ort.InferenceSession(
        quant_weights,
        sess_options=so,
        providers=[provider],
    )

    print(f"Optimized model exported successfully: {output_path}")
    return output_path