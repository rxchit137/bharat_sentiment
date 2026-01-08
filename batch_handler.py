import pandas as pd
import json
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sentiment_logic import analyze
import os
import multiprocessing
from functools import partial

def load_texts_from_file(filepath: str, text_column: str) -> list:
    """Loads text data from a CSV or Excel file."""
    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath)
    elif filepath.endswith('.xlsx'):
        df = pd.read_excel(filepath)
    else:
        raise ValueError("Unsupported file format. Please use CSV or XLSX.")

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in the file.")

    return df[text_column].dropna().tolist()

def extract_keywords(texts: list, n_keywords: int = 10) -> list:
    """Extracts top keywords from a list of texts using TF-IDF."""
    if not texts:
        return []
    vectorizer = TfidfVectorizer(max_features=n_keywords)
    vectorizer.fit_transform(texts)
    return vectorizer.get_feature_names_out().tolist()

def generate_report(filepath: str, texts: list, results: list) -> str:
    """Generates and saves a JSON report from the analysis results."""
    # Overall sentiment and language statistics
    sentiments = [res['sentiment'] for res in results]
    languages = [res['language'] for res in results]
    overall_sentiment_counts = Counter(sentiments)
    language_counts = Counter(languages)

    # Language-specific sentiment statistics
    lang_sentiment_counts = {lang: Counter() for lang in language_counts}
    for res in results:
        lang_sentiment_counts[res['language']][res['sentiment']] += 1

    # Group texts by sentiment for keyword extraction
    texts_by_sentiment = {sentiment: [] for sentiment in overall_sentiment_counts}
    for text, res in zip(texts, results):
        texts_by_sentiment[res['sentiment']].append(text)

    # Extract keywords for each sentiment
    top_keywords = {
        sentiment: extract_keywords(texts_by_sentiment[sentiment])
        for sentiment in texts_by_sentiment
    }

    # Compile the report
    report = {
        "source_file": os.path.basename(filepath),
        "total_texts": len(texts),
        "overall_sentiment_counts": dict(overall_sentiment_counts),
        "language_breakdown": dict(language_counts),
        "language_specific_sentiments": {lang: dict(counts) for lang, counts in lang_sentiment_counts.items()},
        "top_keywords": top_keywords
    }

    # Save the report to a JSON file
    report_filename = f"{os.path.splitext(os.path.basename(filepath))[0]}_report.json"
    report_filepath = os.path.join(os.path.dirname(filepath), report_filename)

    with open(report_filepath, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

    return report_filepath

def compare_reports(report_path1: str, report_path2: str) -> str:
    """Loads two reports, compares them, and saves a comparison report."""
    with open(report_path1, 'r', encoding='utf-8') as f:
        report1 = json.load(f)
    with open(report_path2, 'r', encoding='utf-8') as f:
        report2 = json.load(f)

    # Comparison logic
    comparison = {
        "source_files": [report1["source_file"], report2["source_file"]],
        "total_texts_comparison": {
            "report1": report1["total_texts"],
            "report2": report2["total_texts"],
            "difference": report2["total_texts"] - report1["total_texts"],
        },
        "overall_sentiment_comparison": {},
        "language_breakdown_comparison": {},
        "keyword_comparison": {},
    }

    # Compare overall sentiments
    sentiments1 = Counter(report1["overall_sentiment_counts"])
    sentiments2 = Counter(report2["overall_sentiment_counts"])
    all_sentiments = sorted(list(set(sentiments1.keys()) | set(sentiments2.keys())))
    for sentiment in all_sentiments:
        count1 = sentiments1.get(sentiment, 0)
        count2 = sentiments2.get(sentiment, 0)
        comparison["overall_sentiment_comparison"][sentiment] = {
            "report1": count1,
            "report2": count2,
            "difference": count2 - count1,
        }

    # Compare language breakdown
    langs1 = Counter(report1["language_breakdown"])
    langs2 = Counter(report2["language_breakdown"])
    all_langs = sorted(list(set(langs1.keys()) | set(langs2.keys())))
    for lang in all_langs:
        count1 = langs1.get(lang, 0)
        count2 = langs2.get(lang, 0)
        comparison["language_breakdown_comparison"][lang] = {
            "report1": count1,
            "report2": count2,
            "difference": count2 - count1,
        }

    # Compare keywords
    keywords1 = report1["top_keywords"]
    keywords2 = report2["top_keywords"]
    all_sentiment_keywords = sorted(list(set(keywords1.keys()) | set(keywords2.keys())))
    for sentiment in all_sentiment_keywords:
        k1 = set(keywords1.get(sentiment, []))
        k2 = set(keywords2.get(sentiment, []))
        comparison["keyword_comparison"][sentiment] = {
            "common_keywords": list(k1 & k2),
            "unique_to_report1": list(k1 - k2),
            "unique_to_report2": list(k2 - k1),
        }

    # Save comparison report
    comp_filename = f"comparison_{os.path.splitext(report1['source_file'])[0]}_vs_{os.path.splitext(report2['source_file'])[0]}.json"
    comp_filepath = os.path.join(os.path.dirname(report_path1), comp_filename)

    with open(comp_filepath, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=4)

    return comp_filepath
