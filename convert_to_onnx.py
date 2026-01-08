import os
from optimum.onnxruntime import ORTQuantizer, ORTModel
from optimum.onnxruntime.configuration import AutoQuantizationConfig

def convert_and_quantize_model():
    """
    Converts the PyTorch model to a quantized ONNX model.
    """
    pytorch_model_dir = "./model"
    onnx_output_dir = "./onnx_model"

    if not os.path.exists(pytorch_model_dir) or not os.listdir(pytorch_model_dir):
        print("PyTorch model not found. Please run download_model.py first.")
        return

    print("Loading PyTorch model for conversion...")
    # Load the model from the local directory
    ort_model = ORTModel.from_pretrained(pytorch_model_dir, export=True)

    print("Creating quantization configuration...")
    # Create a quantization configuration that uses dynamic quantization for an optimal trade-off
    # between performance and accuracy on CPU.
    qconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

    print("Creating quantizer...")
    quantizer = ORTQuantizer.from_pretrained(ort_model)

    print("Quantizing model... (This may take a few minutes)")
    # Apply quantization
    quantizer.quantize(save_dir=onnx_output_dir, quantization_config=qconfig)

    print(f"Quantized ONNX model saved to {onnx_output_dir}")

    # Also save the tokenizer to the onnx directory for consistency
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(pytorch_model_dir)
    tokenizer.save_pretrained(onnx_output_dir)
    print(f"Tokenizer saved to {onnx_output_dir}")

if __name__ == "__main__":
    convert_and_quantize_model()
