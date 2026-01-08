import torch
from text_processing import detect_language, transliterate_if_romanized

# Note: The model and tokenizer are now loaded in the worker_init.py module.
# This function is intended to be called by the single-text analysis UI,
# which will need to manage its own model loading.

def analyze(text: str, tokenizer, model) -> dict:
    """
    Analyzes the sentiment of a single text using a pre-loaded tokenizer and model.
    """
    lang = detect_language(text)
    processed_text = transliterate_if_romanized(text, lang)

    inputs = tokenizer(processed_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
    prediction = torch.argmax(probabilities, dim=-1).item()
    sentiment = sentiment_map[prediction]

    return {"language": lang, "sentiment": sentiment}
