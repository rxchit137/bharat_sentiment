import numpy as np
from scipy.special import softmax
from text_processing import detect_language, transliterate_if_romanized

def analyze(text: str, tokenizer, session) -> dict:
    """
    Analyzes the sentiment of a single text using a pre-loaded ONNX session and tokenizer.
    """
    lang = detect_language(text)
    processed_text = transliterate_if_romanized(text, lang)

    # Tokenize and run inference
    inputs = tokenizer(processed_text, return_tensors="np", truncation=True, padding=True, max_length=512)
    ort_inputs = {k: v for k, v in inputs.items()}
    ort_outputs = session.run(None, ort_inputs)

    # Post-process
    probabilities = softmax(ort_outputs[0][0])
    prediction = np.argmax(probabilities)

    sentiment_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
    sentiment = sentiment_map.get(prediction, "unknown")

    return {"language": lang, "sentiment": sentiment}
