from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
from text_processing import detect_language, transliterate_if_romanized

# Global variables to hold the model and tokenizer in each worker process
global_tokenizer = None
global_model = None

def init_worker():
    """
    Initializer function for each worker process in the pool.
    Loads the model and tokenizer into global variables.
    """
    global global_tokenizer, global_model

    model_dir = "./model"
    if not os.path.exists(model_dir) or not os.listdir(model_dir):
        # This check is important to avoid crashing workers if the model is missing.
        # The main thread should handle the user-facing error.
        print(f"Worker process {os.getpid()}: Model not found, cannot initialize.")
        return

    print(f"Worker process {os.getpid()}: Initializing and loading model...")
    global_tokenizer = AutoTokenizer.from_pretrained(model_dir)
    global_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    print(f"Worker process {os.getpid()}: Initialization complete.")

def analyze_text_worker(text: str) -> dict:
    """
    The target function for each worker process.
    Performs sentiment analysis on a single text using the pre-loaded global model.
    """
    global global_tokenizer, global_model

    if global_tokenizer is None or global_model is None:
        return {"language": "unknown", "sentiment": "Worker not initialized; model not loaded."}

    # Detect language and transliterate if necessary
    lang = detect_language(text)
    processed_text = transliterate_if_romanized(text, lang)

    # Perform sentiment analysis
    inputs = global_tokenizer(processed_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = global_model(**inputs)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
    prediction = torch.argmax(probabilities, dim=-1).item()
    sentiment = sentiment_map[prediction]

    return {"language": lang, "sentiment": sentiment}
