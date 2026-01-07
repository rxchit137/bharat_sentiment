from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import os

def analyze(text):
    """
    Analyzes the sentiment of a given Hindi text using a pre-trained model.

    Args:
        text (str): The Hindi text to analyze.

    Returns:
        str: The predicted sentiment (e.g., "Positive", "Negative").
    """
    model_dir = "./model"

    if not os.path.exists(model_dir) or not os.listdir(model_dir):
        return "Model not found. Please run download_model.py first."

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    except Exception as e:
        return f"Error loading model: {e}"


    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_map = {0: "Very Negative", 1: "Negative", 2: "Neutral", 3: "Positive", 4: "Very Positive"}
    prediction = torch.argmax(probabilities, dim=-1).item()

    return sentiment_map[prediction]
