from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

def download_model():
    """Downloads the model and tokenizer and saves them locally."""
    model_name = "tabularisai/multilingual-sentiment-analysis"
    output_dir = "./model"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir)
    print("Tokenizer downloaded successfully.")

    print("Downloading model...")
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.save_pretrained(output_dir)
    print("Model downloaded successfully.")

if __name__ == "__main__":
    download_model()
