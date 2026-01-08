import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from scipy.special import softmax
from text_processing import detect_language, transliterate_if_romanized
from sentiment_logic import analyze

# Global variables for the ONNX session and tokenizer
global_session = None
global_tokenizer = None

def init_worker():
    """
    Initializer for each worker process. Loads the ONNX model and tokenizer.
    """
    global global_session, global_tokenizer

    model_dir = "./onnx_model"
    model_path = os.path.join(model_dir, "model_quantized.onnx")

    if not os.path.exists(model_path):
        print(f"Worker {os.getpid()}: ONNX model not found, cannot initialize.")
        return

    print(f"Worker {os.getpid()}: Initializing ONNX session...")
    global_session = ort.InferenceSession(model_path)
    global_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    print(f"Worker {os.getpid()}: Initialization complete.")

def analyze_text_worker(text: str) -> dict:
    """
    Target function for worker processes to perform analysis using ONNX.
    """
    global global_session, global_tokenizer

    if global_session is None or global_tokenizer is None:
        return {"language": "unknown", "sentiment": "Worker not initialized; ONNX model not loaded."}

    return analyze(text, global_tokenizer, global_session)
