import streamlit as st

from predictor import load_artifacts, predict_news

st.set_page_config(page_title="Fake News Detection", page_icon="📰", layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def get_artifacts():
    return load_artifacts()


try:
    artifacts = get_artifacts()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

st.title("📰 Fake News Detection")
st.write("Enter a news headline and optional source information to classify it as Real or Fake.")
st.info(
    "This model learns patterns from the supplied Fake_News_Net dataset. "
    "It does not independently verify facts or read the complete article."
)
st.divider()

col1, col2 = st.columns([2, 1])
with col1:
    title = st.text_area("News Headline", placeholder="Enter the news headline here...", height=150)
    source_domain = st.text_input("Source Domain", placeholder="example.com")
with col2:
    tweet_num = st.number_input("Tweet / Share Count", min_value=0, value=0, step=1)
    st.write("")
    st.write("")
    predict_button = st.button("🔍 Predict News", use_container_width=True)

if predict_button:
    if not title.strip():
        st.warning("Please enter a news headline.")
    else:
        with st.spinner("Analyzing news..."):
            result = predict_news(artifacts, title, source_domain, tweet_num)

        st.divider()
        if result["prediction"] == "REAL":
            st.success("🟢 REAL NEWS")
        else:
            st.error("🔴 FAKE NEWS")

        st.subheader("Prediction Details")
        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction", result["prediction"])
        c2.metric("Real Probability", f"{result['probability_real'] * 100:.2f}%")
        c3.metric("Fake Probability", f"{result['probability_fake'] * 100:.2f}%")

        st.write("### Confidence")
        st.progress(min(max(result["confidence"], 0.0), 1.0))
        st.write(f"Model confidence: **{result['confidence'] * 100:.2f}%**")
