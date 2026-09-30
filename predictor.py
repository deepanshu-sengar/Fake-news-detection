"""Model loading and prediction (framework-independent, easy to test)."""
from pathlib import Path

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack

from text_utils import clean_text

MODEL_PATH = Path(__file__).resolve().parent / "models" / "fake_news_detector.joblib"


def load_artifacts(path: Path = MODEL_PATH) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found at {path}. Run `python train.py` first.")
    return joblib.load(path)


def predict_news(artifacts: dict, title, source_domain="", tweet_num=0) -> dict:
    title = "" if title is None else str(title)
    source_domain = str(source_domain or "").lower().strip() or "unknown"
    tweet_num = max(0.0, float(tweet_num or 0))

    title_clean = clean_text(title)
    features = hstack([
        artifacts["word_vectorizer"].transform([title_clean]),
        artifacts["char_vectorizer"].transform([title_clean]),
        artifacts["domain_encoder"].transform([[source_domain]]),
        csr_matrix(artifacts["tweet_scaler"].transform(np.log1p([[tweet_num]]))),
    ]).tocsr()

    p_real = float(artifacts["model"].predict_proba(features)[0, 1])
    p_fake = 1.0 - p_real
    label = "REAL" if p_real >= 0.5 else "FAKE"
    return {
        "prediction": label,
        "probability_real": p_real,
        "probability_fake": p_fake,
        "confidence": p_real if label == "REAL" else p_fake,
    }
