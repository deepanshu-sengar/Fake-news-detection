"""Train the fake-news classifier and save it to models/.

Usage:  python train.py
"""
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             classification_report, f1_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from text_utils import clean_text

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Fake_News_Net.csv"
MODEL_PATH = BASE_DIR / "models" / "fake_news_detector.joblib"


def main():
    df = pd.read_csv(DATA_PATH)
    df["title"] = df["title"].fillna("").astype(str)
    df["source_domain"] = df["source_domain"].fillna("unknown").astype(str).str.lower().str.strip()
    df["tweet_num"] = pd.to_numeric(df["tweet_num"], errors="coerce").fillna(0)
    df["real"] = pd.to_numeric(df["real"], errors="coerce")
    df = df[df["real"].isin([0, 1])].copy()
    df["title_clean"] = df["title"].apply(clean_text)
    df = df.drop_duplicates(subset=["title_clean", "source_domain", "tweet_num", "real"]).reset_index(drop=True)
    print("Rows after cleaning:", len(df))

    X_train, X_test, y_train, y_test = train_test_split(
        df[["title_clean", "source_domain", "tweet_num"]], df["real"],
        test_size=0.20, random_state=RANDOM_STATE, stratify=df["real"])

    word_vectorizer = TfidfVectorizer(sublinear_tf=True, strip_accents="unicode", min_df=2,
                                      max_df=0.98, ngram_range=(1, 2), max_features=120000)
    char_vectorizer = TfidfVectorizer(analyzer="char", sublinear_tf=True, min_df=2,
                                      max_features=100000, ngram_range=(3, 5))
    domain_encoder = OneHotEncoder(handle_unknown="ignore")
    tweet_scaler = StandardScaler()

    def build(frame, fit):
        f = (lambda o, x: o.fit_transform(x)) if fit else (lambda o, x: o.transform(x))
        return hstack([
            f(word_vectorizer, frame["title_clean"]),
            f(char_vectorizer, frame["title_clean"]),
            f(domain_encoder, frame[["source_domain"]]),
            csr_matrix(f(tweet_scaler, np.log1p(frame[["tweet_num"]]))),
        ]).tocsr()

    X_tr, X_te = build(X_train, True), build(X_test, False)

    model = LogisticRegression(C=2.0, max_iter=2000, class_weight="balanced",
                               solver="liblinear", random_state=RANDOM_STATE)
    model.fit(X_tr, y_train)

    pred = model.predict(X_te)
    prob = model.predict_proba(X_te)[:, 1]
    print(f"Accuracy          : {accuracy_score(y_test, pred):.4f}")
    print(f"Balanced accuracy : {balanced_accuracy_score(y_test, pred):.4f}")
    print(f"F1                : {f1_score(y_test, pred):.4f}")
    print(f"ROC-AUC           : {roc_auc_score(y_test, prob):.4f}")
    print(classification_report(y_test, pred, target_names=["Fake", "Real"], digits=4))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    # clean_text is intentionally NOT pickled; it is imported from text_utils.
    joblib.dump({
        "model": model,
        "word_vectorizer": word_vectorizer,
        "char_vectorizer": char_vectorizer,
        "domain_encoder": domain_encoder,
        "tweet_scaler": tweet_scaler,
    }, MODEL_PATH, compress=3)
    print("Saved:", MODEL_PATH)


if __name__ == "__main__":
    main()
