"""Text preprocessing shared by training and inference.

Lives in its own module (not in a notebook / __main__) so the saved model
can be loaded reliably from any entry point (Streamlit, Docker, scripts).
"""
import re


def clean_text(text) -> str:
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
