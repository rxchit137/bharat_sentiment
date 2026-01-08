import os
import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer
from scipy.special import softmax
from text_processing import detect_language, transliterate_if_romanized

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

    lang = detect_language(text)
    processed_text = transliterate_if_romanized(text, lang)

    # Tokenize input
    inputs = global_tokenizer(processed_text, return_tensors="np", truncation=True, padding=True, max_length=512)

    # Run inference with ONNX Runtime
    ort_inputs = {k: v for k, v in inputs.items()}
    ort_outputs = global_session.run(None, ort_inputs)

    # Post-process the output
    probabilities = softmax(ort_outputs[0][0])
    prediction = np.argmax(probabilities)

    sentiment_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
    sentiment = sentiment_map[prediction.item()]

    return {"language": lang, "sentiment": sentiment}
