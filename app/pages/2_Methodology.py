import streamlit as st

st.set_page_config(
    page_title="Methodology",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 Methodology")
st.caption("How the misinformation model was built and evaluated.")

st.markdown("## 1. Research Question")
st.markdown(
    """
This project asks:

**Where does predictive signal in misinformation detection come from: textual content or diffusion-based features?**

Rather than only optimizing a classifier, the goal is to compare **text-only**, **metadata-only**, and **hybrid** approaches to understand which signals actually drive predictive performance.
"""
)

st.markdown("---")

st.markdown("## 2. Pipeline Overview")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("### Problem")
    st.write("Identify how misinformation can be detected and which feature sources matter most.")

with col2:
    st.markdown("### Data")
    st.write("Use labeled misinformation datasets including FakeNewsNet PolitiFact, ISOT, and a COVID metadata dataset.")

with col3:
    st.markdown("### Features")
    st.write("Extract TF-IDF text features, metadata variables, and hybrid combinations of both.")

with col4:
    st.markdown("### Model")
    st.write("Train interpretable Logistic Regression classifiers on sparse feature representations.")

with col5:
    st.markdown("### Evaluation")
    st.write("Compare models using precision, recall, F1, PR curves, and threshold tuning.")

st.markdown("---")

st.markdown("## 3. Dataset Used in the Demo")
st.markdown(
    """
The interactive demo is powered by the **FakeNewsNet PolitiFact text model**.

This dataset contains:
- news articles labeled as **real** or **fake**
- political/news-style writing
- text suitable for TF-IDF based classification

This model was chosen for the demo because it is the most realistic **text-only deployment setting**: a user can paste article text directly and receive a prediction.
"""
)

st.markdown("---")

st.markdown("## 4. Feature Engineering")
st.markdown(
    """
### Text Features
- article text was cleaned and processed
- converted into **TF-IDF vectors**
- used **unigrams + bigrams**
- emphasized informative terms while downweighting common terms

### Why TF-IDF?
TF-IDF is a strong baseline for text classification because it captures which words and phrases are distinctive in each document while remaining computationally efficient and interpretable.
"""
)

st.markdown("---")

st.markdown("## 5. Model Choice")
st.markdown(
    """
The demo uses **Logistic Regression** on top of TF-IDF features.

### Why Logistic Regression?
- strong baseline for sparse text data
- fast to train and deploy
- interpretable coefficients
- enables feature-level explanations in the demo

This makes it well-suited for a project that emphasizes both **predictive performance** and **interpretability**.
"""
)

st.markdown("---")

st.markdown("## 6. Evaluation Strategy")
st.markdown(
    """
Model performance was evaluated using:

- **Precision**
- **Recall**
- **F1 Score**
- **Precision–Recall analysis**
- **Threshold tuning**

This matters because misinformation detection is an **imbalanced classification problem**, so accuracy alone can be misleading.
"""
)

st.info("Next: the Results page will compare text, metadata, and hybrid models directly.")