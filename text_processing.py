from langdetect import detect, LangDetectException
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

def detect_language(text: str) -> str:
    """
    Detects the language of a given text snippet.
    Returns the ISO 639-1 language code (e.g., 'en', 'hi').
    Returns 'unknown' if detection fails.
    """
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"

def transliterate_if_romanized(text: str, lang: str) -> str:
    """
    Transliterates the text to Devanagari if it is likely Romanized Hindi.
    A simple heuristic is used: if the detected language is 'en',
    transliteration is attempted.
    """
    if lang == 'en':
        # This heuristic assumes that English text might be Romanized Hindi.
        # In a real-world scenario, a more sophisticated check would be needed.
        transliterated_text = transliterate(data=text, _from=sanscript.ITRANS, _to=sanscript.DEVANAGARI)
        # Return the transliterated text only if it's different from the original
        # and contains Devanagari characters, otherwise return the original text.
        if transliterated_text != text and any(ord(c) > 127 for c in transliterated_text):
            return transliterated_text
    return text
