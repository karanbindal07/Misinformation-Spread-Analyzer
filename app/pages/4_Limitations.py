import streamlit as st

st.set_page_config(
    page_title="Limitations",
    page_icon="⚠️",
    layout="wide",
)

st.title("⚠️ Limitations & Future Work")
st.caption("What this project captures well, what it misses, and how it can be improved.")

st.markdown("## 1. Why limitations matter")

st.markdown(
    """
A strong machine learning project should not only show **what works**, but also explain **what remains unresolved**.

This project performs well as an interpretable baseline for misinformation detection, but several important limitations remain.
"""
)

st.markdown("---")

st.markdown("## 2. Current limitations")

st.markdown(
    """
### 1. TF-IDF does not understand deep semantic meaning
The model relies on word and phrase frequency patterns rather than true contextual understanding.
It can identify stylistic cues, but it does not reason about factuality or meaning the way modern transformer-based models can.

### 2. Token-level explanations can be noisy
Because the model is based on bag-of-words style features, some explanations reflect tokenization artifacts or highly local lexical patterns rather than broader narrative meaning.

### 3. Metadata modeling is limited
The metadata-only benchmark uses lightweight structured features rather than full propagation networks.
As a result, it does not capture richer social diffusion dynamics such as repost cascades, user interaction structure, or temporal spread patterns.

### 4. Dataset dependence affects generalization
The FakeNewsNet PolitiFact benchmark is politically oriented, which means the learned patterns may not fully generalize to other forms of misinformation such as health rumors, conspiracy narratives, or manipulated multimedia content.

### 5. Classification is not the same as truth verification
This model predicts whether text resembles patterns associated with misinformation in the training data.
It does **not** independently verify claims against ground truth sources.
"""
)

st.markdown("---")

st.markdown("## 3. Practical implications")

st.markdown(
    """
These limitations mean the system should be interpreted as a **risk triage tool**, not a fully autonomous fact-checking engine.

In practice, the model is best used for:

- prioritizing suspicious content for review
- supporting analysts with interpretable signals
- comparing feature sources in misinformation detection experiments

It should **not** be treated as a final authority on whether a claim is true or false.
"""
)

st.markdown("---")

st.markdown("## 4. Future work")

st.markdown(
    """
Several improvements would make the project stronger both technically and practically:

### 1. Transformer-based models
Replacing TF-IDF with models such as BERT or DeBERTa would improve contextual understanding and potentially strengthen cross-domain generalization.

### 2. Richer propagation features
A stronger metadata benchmark could include temporal spread, user interaction graphs, repost depth, and community-level diffusion structure.

### 3. Cross-domain evaluation
Testing the same model on political misinformation, conspiracy narratives, and health misinformation would provide a better understanding of robustness.

### 4. Better explainability tools
Adding SHAP values, phrase-level attributions, or attention-based explanations could improve interpretability beyond token-level contributions.

### 5. Human-in-the-loop workflows
The most realistic deployment would integrate the model into a review pipeline where analysts validate predictions rather than relying on fully automated decisions.
"""
)

st.markdown("---")

st.markdown("## 5. Final reflection")

st.success(
    "This project demonstrates that interpretable machine learning can provide useful signal for misinformation risk triage, while also showing why robust detection requires stronger context modeling, richer diffusion features, and careful human oversight."
)