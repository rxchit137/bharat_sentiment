import tkinter as tk
from tkinter import messagebox
from sentiment_logic import analyze

class SentimentApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hindi Sentiment Analysis")

        self.label = tk.Label(root, text="Enter Hindi text:")
        self.label.pack(pady=10)

        self.text_entry = tk.Entry(root, width=50)
        self.text_entry.pack(pady=10, padx=10)

        self.analyze_button = tk.Button(root, text="Analyze", command=self.analyze_sentiment)
        self.analyze_button.pack(pady=10)

        self.result_label = tk.Label(root, text="Sentiment: ")
        self.result_label.pack(pady=10)

    def analyze_sentiment(self):
        text = self.text_entry.get()
        if not text:
            messagebox.showwarning("Input Error", "Please enter some text.")
            return

        sentiment = analyze(text)
        self.result_label.config(text=f"Sentiment: {sentiment}")

if __name__ == "__main__":
    root = tk.Tk()
    app = SentimentApp(root)
    root.mainloop()
