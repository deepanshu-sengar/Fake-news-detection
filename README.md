# 📰 Fake News Detection

A Streamlit web app that classifies a news headline as **REAL** or **FAKE** using a
Logistic Regression model trained on the Fake_News_Net dataset (word + character TF-IDF,
source-domain one-hot, log tweet count).

> The model only learns patterns from headlines and metadata. It does **not** fact-check.

## Project structure

```
app.py            Streamlit UI
predictor.py      Model loading + prediction
text_utils.py     Text cleaning (shared by training and inference)
train.py          Retrains the model from data/Fake_News_Net.csv
models/           Trained model (fake_news_detector.joblib)
data/             Training dataset
notebooks/        Exploratory notebook
requirements.txt  Pinned dependencies
```

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Retrain (optional)

```bash
python train.py
```

Retrain whenever you change the `scikit-learn` version, since pickled models are version-specific.

## Deploy

**Streamlit Community Cloud:** push to GitHub, create a new app, set main file to `app.py`.


